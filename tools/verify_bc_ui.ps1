# tools/verify_bc_ui.ps1
#
# Verifies the IN-MATCH behavior-cloning option in the real GUI (2026-10):
#   pick A = a BC-capable agent, B = a NON-Alpha-Beta opponent (MCTS)  ->  the
#   "行为克隆训练" teacher DROPDOWN becomes available  ->  pick Alpha-Beta depth 1 in it
#   ->  play one game  ->  the student's own positions really become supervision samples
#   AND really train its policy head  ->  the per-match summary is in the panel  ->  and
#   NOTHING was written to the standard weight paths (BC only touches memory; the only
#   save point is exit).
#
# User requirement this pins down (verbatim, 2026-10): "行为克隆勾选框改成下拉框选择要克隆
# 的 abagent，与将要对弈的对方 agent 或者人类棋手无关，训练的时候参考下拉框选择的 abagent
# 的决策进行行为克隆训练". Two consequences this script asserts explicitly:
#   * the teacher comes from the DROPDOWN, not from the opponent -- the opponent here is
#     MCTS, i.e. there is no Alpha-Beta anywhere on the board;
#   * the old checkbox must be GONE (a UI that still has it is a mix of two口径).
# The earlier requirement ("no standalone 'start behavior cloning' button") is still checked.
#
# Why this script exists (what ctest cannot cover):
#   `test_bc` pins the CALIPER of BC (gradients, masked CE, "critic untouched", ...) but
#   it never touches the GUI path: the dropdown, its enable rule, the match hook, the
#   panel plumbing. Any of those can be broken while every ctest still passes.
#
# Method: UIA only, ASCII only. Why ASCII matters: a BOM-less UTF-8 .ps1 is decoded as
# ANSI by Windows PowerShell, so a CJK literal inside CODE breaks the parser (comments
# are fine). Every assertion therefore matches ASCII tokens that the product itself
# prints -- the per-match summary line carries "match-summary: samples= updates=
# targetMissed=" precisely so that a script can find it without reading CJK.
#
# Side effect (on purpose): this launches the REAL chess.exe from the build directory and
# lets it play a real (short) game. It kills that process instead of closing it, so the
# app's "save every agent on exit" never fires -- this script must not touch the weights
# the user already has.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\verify_bc_ui.ps1
#   powershell -ExecutionPolicy Bypass -File tools\verify_bc_ui.ps1 -MatchTimeoutSec 600
#   :: [2026-10 user question] "勾选行为克隆训练后能否勾选 rollout 预训练":
#   powershell -ExecutionPolicy Bypass -File tools\verify_bc_ui.ps1 -KeepPreTrain
param(
    [string]$Exe = "",
    [int]$TimeoutSec = 300,
    # A full game can take a couple of minutes (the learner thinks on every move).
    [int]$MatchTimeoutSec = 420,
    # [2026-10] Keep "探索+预训练" (per-move rollout pretraining) ON instead of switching it
    # off. The default path turns it OFF only to keep this check short (pretraining makes every
    # move several times slower) -- but that also means the default run never exercises the
    # "BC + pretraining at the same time" combination. This switch is that run: it asserts the
    # pretraining checkbox stays ENABLED and ON while a BC teacher is selected (the direct
    # answer to the user's question) and then asserts BC still accumulates samples/updates.
    [switch]$KeepPreTrain,
    # ASCII substrings that pick the A/B rows. The trailing comma in the student token is
    # what makes it unambiguous: the TB variant is "PPO+MCTS (AlphaZero)" and would match
    # the shorter prefix too. The MLP variant is the cheap one (~25x) -- and it is the one
    # whose self-check line this script also reads.
    [string]$StudentMatch = "PPO+MCTS (AlphaZero,",
    # [2026-10] The OPPONENT is deliberately a NON-Alpha-Beta agent (MCTS 800). That is the
    # whole point of the new semantics ("与将要对弈的对方 agent ... 无关"): the teacher comes
    # from the dropdown, so BC must run even when nobody on the board is an AB.
    [string]$OpponentMatch = "MCTS (800",
    # Teacher selected in the dropdown (its item text carries the ASCII marker "depth=N").
    # [2026-10] Default is depth=2, and that is deliberate: the soft-target switch is exercised
    # by this run, and with depth=1 the soft target **degenerates to one-hot** (there is only
    # one depth to vote, so H(t)=0 -- the product now says so in the report). Running at
    # depth=2 is what makes `targetH>0` a meaningful assertion instead of a tautology.
    [string]$TeacherItemMatch = "depth=2",
    # Negative control: a pair with NO BC-capable student at all (pure search vs pure
    # search) => the dropdown must stay disabled. Note this is NOT "no AB opponent" any
    # more: under the 2026-10 semantics an AB opponent is irrelevant.
    [string]$NoStudentAMatch = "MCTS (800",
    [string]$NoStudentBMatch = "Alpha-Beta L1",
    # A pure-search agent for the human-game AI row (see the negative-control comment).
    [string]$PureSearchMatch = "Alpha-Beta Pruning",
    # [2026-10] 人机那一段的两个刻度 (都是"脚本怎么打这一局", 与产品口径无关):
    #  - MoveSettleMs: 点完一对 from/to 之后等多久再继续。**故意偏短**: 那一步不合法的
    #    时候不会有任何反应, 而"等得久"只会让整段变慢 —— 真正的判据在下面对面板文本的
    #    切片里 (见 human game 那一段的注释), 不依赖逐手识别。
    #  - HumanPhaseSec: 整段人机对弈的时间预算 (含"AI 把盲走的红方将死 -> 关弹窗重开一局")。
    [int]$MoveSettleMs = 1200,
    # [2026-10 user 口径] "不进行人机对弈测试": the human section drives REAL clicks on the board
    # and plays a real game against the AI on screen (the user watches themselves get checked /
    # mated). That is not always wanted, so the section has an off switch. Skipping it does not
    # silently pass anything -- it produces NO assertions for that section at all, and the run
    # reports it as skipped so nobody reads the missing coverage as a pass.
    [switch]$SkipHuman,
    # 预算默认给得**不长** (75 s): 这一段会在屏幕上真下一局棋, 而脚本是盲走红兵 —— 用户会
    # 看着自己被将军/将死。确定性断言(建线 + BC 跑过)一秒内就能拿到, 长的部分只是"多熬几手
    # 看曲线出点", 不值得让人盯着屏幕。要那一条就把这个值调大。
    [int]$HumanPhaseSec = 75
)

$ErrorActionPreference = "Stop"
# The number the product prints in its machine-readable markers (`[teacher-depth=N]` in the
# report header, `[teacher-depth=N soft=M]` in the live line). Derived from the item text above
# so the two can never drift apart (that drift is the classic "the script asserts depth=1 while
# it selected depth=2" false failure). There is no closing bracket in the live-line pattern
# because `soft=` now follows it in the same bracket.
$TeacherDepth = [int]([regex]::Match($TeacherItemMatch, "[0-9]+").Value)
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path
$exeDir = Split-Path $Exe -Parent

$script:fail = 0
function Check([bool]$ok, [string]$what, [string]$detail = "") {
    if ($ok) {
        Write-Host ("  PASS  " + $what + $(if ($detail -ne "") { "  (" + $detail + ")" } else { "" }))
    } else {
        Write-Host ("  FAIL  " + $what + $(if ($detail -ne "") { "  (" + $detail + ")" } else { "" })) -ForegroundColor Red
        $script:fail++
    }
}

function PropOf($e, $p) {
    try {
        $v = $e.GetCurrentPropertyValue($p)
        if ($v -eq [System.Windows.Automation.AutomationElement]::NotSupported) { return "" }
        return [string]$v
    } catch { return "" }
}

# Qt's Windows UIA bridge fills AutomationId with the QObject objectName path
# ("QApplication.MainWindow.centralwidget.controlWidget.matchAComboBox"), so matching on
# the tail is language independent (same idiom as verify_ppo_moe_selfcheck.ps1).
function FindByObjectName($root, [string]$objectName) {
    foreach ($e in (AllDescendants $root)) {
        $aid = PropOf $e ([System.Windows.Automation.AutomationElement]::AutomationIdProperty)
        if ($aid.EndsWith("." + $objectName)) { return $e }
    }
    return $null
}

function AllDescendants($root) {
    # Every UIA call here is wrapped: Qt recreates/removes some widgets while a match runs
    # (and popups come and go), so "the target element is no longer available" is a normal
    # transient -- it must not abort the script (the first version threw out of the tree
    # walk and reported nothing).
    $out = New-Object System.Collections.ArrayList
    $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
    $stack = New-Object System.Collections.Stack
    $stack.Push($root)
    while ($stack.Count -gt 0) {
        $cur = $stack.Pop()
        [void]$out.Add($cur)
        try {
            $ch = $walker.GetFirstChild($cur)
            while ($ch -ne $null) {
                $stack.Push($ch)
                $ch = $walker.GetNextSibling($ch)
            }
        } catch { continue }
    }
    return $out
}

function Enabled($e) {
    if ($e -eq $null) { return $false }
    try { return [bool]$e.Current.IsEnabled } catch { return $false }
}

# The self-check panel: a multi-line Edit whose ValuePattern carries the whole text.
function PanelText($root) {
    foreach ($e in (AllDescendants $root)) {
        $ct = PropOf $e ([System.Windows.Automation.AutomationElement]::ControlTypeProperty)
        if ($ct -ne [string][System.Windows.Automation.ControlType]::Edit) { continue }
        $vp = $null
        try { $vp = $e.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern) } catch { continue }
        if ($vp -eq $null) { continue }
        $t = $vp.Current.Value
        if ($t -ne $null -and $t.Length -gt 80 -and
            ($t -match "prior top-1|PPO|SAC|MoE|match-summary|selfcheck|agent")) {
            return $t
        }
    }
    return ""
}

# Strict substring search over every Edit control. Why this exists in addition to
# PanelText: PanelText has heuristics (minimum length + a token list) that a panel in a
# transient state can fail -- during a match the panel holds only the lines this feature
# appended, and the first version of this script then waited out its whole window and
# reported FAIL while the text it wanted was one poll away. A needle search has no
# heuristics to be wrong about.
function EditTextContaining($root, [string]$needle) {
    foreach ($e in (AllDescendants $root)) {
        try {
            $vp = $e.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
            if ($vp -eq $null) { continue }
            $t = $vp.Current.Value
            if ($t -ne $null -and $t.Contains($needle)) { return $t }
        } catch { continue }
    }
    return ""
}

# Qt popups ignore SelectionItemPattern, so click the item for real (the fix
# verify_chess_saves_weights.ps1 documents).
function Select-ComboItemByMatch($root, [string]$objectName, [string]$match) {
    $combo = FindByObjectName $root $objectName
    if ($combo -eq $null) { return "" }
    $ec = $null
    if (-not $combo.TryGetCurrentPattern(
            [System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$ec)) {
        return ""
    }
    $ec.Expand()
    Start-Sleep -Milliseconds 700
    $liCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    $items = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $liCond)
    $target = $null
    $name = ""
    foreach ($it in $items) {
        $n = PropOf $it ([System.Windows.Automation.AutomationElement]::NameProperty)
        if ($n -like ("*" + $match + "*")) { $target = $it; $name = $n; break }
    }
    if ($target -ne $null) {
        $r = $target.Current.BoundingRectangle
        if ($r.Width -gt 0 -and $r.Height -gt 0) {
            [System.Windows.Forms.Cursor]::Position =
                New-Object System.Drawing.Point([int]($r.X + $r.Width / 2), [int]($r.Y + $r.Height / 2))
            Start-Sleep -Milliseconds 120
            Add-Type -MemberDefinition '[DllImport("user32.dll")]public static extern void mouse_event(int f,int x,int y,int d,int e);' -Name Ubc -Namespace Wbc -PassThru | Out-Null
            [Wbc.Ubc]::mouse_event(0x0002, 0, 0, 0, 0)
            [Wbc.Ubc]::mouse_event(0x0004, 0, 0, 0, 0)
            Start-Sleep -Milliseconds 400
        }
    }
    $ec.Collapse()
    Start-Sleep -Milliseconds 300
    return $name
}

# Snapshot every ListItem on the desktop as "runtime id -> name". Used to diff the tree
# across an Expand(): the popup's items are exactly the ones that appear in between.
function ListItemMap {
    $map = @{}
    $liCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    try {
        foreach ($it in [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
                [System.Windows.Automation.TreeScope]::Descendants, $liCond)) {
            try {
                $id = ($it.GetRuntimeId() -join "_")
                $map[$id] = PropOf $it ([System.Windows.Automation.AutomationElement]::NameProperty)
            } catch { }
        }
    } catch { }
    return $map
}

# Read every item of a combo. Qt creates the popup (and therefore the item elements)
# LAZILY: with the combo closed, FindAll(Descendants) on it returns **0** items (measured
# on this build -- the first version of this helper printed an empty list and two checks
# failed while the dropdown itself was perfectly fine).
#
# So: expand, then take the **diff** of the desktop's ListItems across the expand. The diff
# matters -- just enumerating everything visible while a popup is open also picks up the
# app's own list widgets (the per-game list, panel texts...), which is how the first working
# version printed "关 (off) | ... | (还没有对局) | <build directory listing>" as if all of
# that were the dropdown's items.
function ComboItemNames($combo) {
    $out = New-Object System.Collections.ArrayList
    if ($combo -eq $null) { return $out }
    $ec = $null
    if (-not $combo.TryGetCurrentPattern(
            [System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$ec)) {
        return $out
    }
    $before = ListItemMap
    $ec.Expand()
    Start-Sleep -Milliseconds 700
    $after = ListItemMap
    $ec.Collapse()
    Start-Sleep -Milliseconds 300
    foreach ($k in $after.Keys) {
        if (-not $before.ContainsKey($k)) { [void]$out.Add($after[$k]) }
    }
    return $out
}

# ---- 人机那一段用: 真的把鼠标点到棋盘上 (UIA 点不动自绘控件) ----
# ChessBoard 的几何是写死的私有常量 (offsetX/offsetY = 50, gridSize = 60, 见 chessboard.h),
# 与 verify_human_vs_ai.ps1 / probe_hvai_flow 用的是同一套映射: (col,row) -> 屏幕坐标。
function BcBoardPoint([int]$col, [int]$row) {
    return @(($script:bcBoardLeft + 50 + $col * 60), ($script:bcBoardTop + 50 + $row * 60))
}
function BcClick([int]$x, [int]$y) {
    Add-Type -MemberDefinition '[DllImport("user32.dll")]public static extern bool SetCursorPos(int x,int y); [DllImport("user32.dll")]public static extern void mouse_event(int f,int x,int y,int d,int e);' -Name BcM -Namespace BcW -PassThru | Out-Null
    [BcW.BcM]::SetCursorPos($x, $y) | Out-Null
    Start-Sleep -Milliseconds 80
    [BcW.BcM]::mouse_event(0x0002, 0, 0, 0, 0)
    [BcW.BcM]::mouse_event(0x0004, 0, 0, 0, 0)
}

# ---- 人机那一段用: 终局弹窗 ----
# 脚本盲走红兵**会被 AI 将死**(实测: 用户在自己的屏幕上看到了 "黑方胜!")。弹窗是
# exec() 起的模态嵌套事件循环, 不关掉它后面的点击全都被吃掉, 而且人机那条路的读数与曲线
# 都是**每局**口径 —— 一局凑不满 8 手就只能关掉弹窗、开新局再来。
# 三个标题是产品自己弹的 (与 verify_human_vs_ai.ps1 用的是同一份字面量)。
function Find-BcEndDialog {
    foreach ($n in @("黑方胜!", "红方胜!", "平局!")) {
        $e = Find-ByName $n
        if ($e -ne $null) { return $e }
    }
    return $null
}

# 按 Name (精确) 找控件。FindByObjectName 是按 AutomationId 找的, 而 QMessageBox 的
# 标题就是它的 Name —— 两种查找各管一路, 都需要。
function Find-ByName([string]$name) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::IsControlElementProperty, $true)
    foreach ($e in (AllDescendants $root)) {
        try {
            if ([string]$e.Current.Name -eq $name) { return $e }
        } catch { }
    }
    return $null
}

function Set-Spin($root, [string]$objectName, [double]$value) {    $sp = FindByObjectName $root $objectName
    if ($sp -eq $null) { return $false }
    try {
        $pat = $null
        if (-not $sp.TryGetCurrentPattern(
                [System.Windows.Automation.RangeValuePattern]::Pattern, [ref]$pat)) {
            return $false
        }
        $pat.SetValue($value)
        Start-Sleep -Milliseconds 150
        return $true
    } catch {
        # A disabled spin box throws here ("operation is not allowed on a nonenabled
        # element") -- report it as false instead of aborting the whole script, so the
        # remaining assertions still produce a readable report.
        Write-Host ("  info  SetValue failed on " + $objectName + ": " + $_.Exception.Message)
        return $false
    }
}

function ToggleStateOf($e) {
    if ($e -eq $null) { return "none" }
    try {
        $tp = $e.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
        return [string]$tp.Current.ToggleState
    } catch { return "none" }
}

# Standard weight files = everything in weights/ that is NOT a "weights/_temp*" transient
# (those are diverted to memory during matches; BC writes neither kind).
function StandardWeightSnapshot([string]$dir) {
    $w = Join-Path $dir "weights"
    if (-not (Test-Path $w)) { return @() }
    return @(Get-ChildItem $w -File -ErrorAction SilentlyContinue |
             Where-Object { $_.Name -notlike "_temp*" } |
             ForEach-Object { $_.Name + "|" + $_.Length + "|" + $_.LastWriteTimeUtc.Ticks })
}

Write-Host "=== verify_bc_ui (in-match behavior cloning) ==="
Write-Host ("exe: " + $Exe)

$weightsBefore = StandardWeightSnapshot $exeDir
Write-Host ("standard weight files before: " + $weightsBefore.Count)

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
try {
    # ---- 1. window + UI built ----
    $dl = (Get-Date).AddSeconds($TimeoutSec)
    $root = $null
    while ((Get-Date) -lt $dl) {
        Start-Sleep -Milliseconds 500
        if ($proc.HasExited) { break }
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
        $wins = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Children, $cond)
        if ($wins.Count -gt 0) { $root = $wins[0]; break }
    }
    Check ($root -ne $null) "main window found via UIA"
    if ($root -eq $null) { throw "no main window" }

    $combo = $null
    $dl0 = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $dl0) {
        Start-Sleep -Milliseconds 1000
        if ($proc.HasExited) { break }
        $combo = FindByObjectName $root "matchAComboBox"
        if ($combo -ne $null) { break }
    }
    Check ($combo -ne $null) "matchAComboBox found (UI finished building)"
    if ($combo -eq $null) { throw "no match combo" }

    # ---- 1b. wait for the STARTUP LOAD to finish ----
    # Why this is not optional: during the preload the app disables every interactive
    # control (including this dropdown and the game-count spin box), and the combos can be
    # expanded but the enable state is decided again only in the `startupComplete` handler.
    # Asserting on the dropdown before that produced a false FAIL in the first version of
    # this script ("disabled for student + AB teacher") -- the wiring was fine,
    # the app was simply still loading 9 weight groups (~40 s on this machine).
    # `agentComboBox` is the documented "startup finished" probe (re-enabled there).
    $started = $false
    $dlStart = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $dlStart) {
        Start-Sleep -Milliseconds 1000
        if ($proc.HasExited) { break }
        if (Enabled (FindByObjectName $root "agentComboBox")) { $started = $true; break }
    }
    Check $started "startup weight load finished (agentComboBox re-enabled)"
    if (-not $started) { throw "startup never finished" }

    # ---- 2. the teacher dropdown exists, and the OLD entry points are gone ----
    # [2026-10 user wording] The checkbox was replaced by a dropdown that picks WHICH
    # Alpha-Beta to clone. Two legacy things must be gone: the standalone "run BC" button,
    # and the checkbox itself (if it is still there, the UI is a mix of two口径).
    $bcCmb = FindByObjectName $root "bcTeacherCombo"
    Check ($bcCmb -ne $null) "bcTeacherCombo exists (the teacher dropdown)"
    Check ((FindByObjectName $root "bcInMatchCheck") -eq $null) `
        "the old checkbox is gone (replaced by the teacher dropdown)"
    Check ((FindByObjectName $root "bcStartBtn") -eq $null) `
        "no standalone 'start behavior cloning' button any more (user requirement)"
    # [2026-10] The soft-target switch sits NEXT TO the dropdown (it changes what the teacher
    # hands over -- one move vs a distribution -- not a separate feature), so it must exist and
    # must start OFF (the A/B experiment showed no holdout gain; see docs section 9).
    $softChk = FindByObjectName $root "bcSoftCheck"
    Check ($softChk -ne $null) "bcSoftCheck exists (the soft-target switch next to the dropdown)"
    if ($softChk -ne $null) {
        Check ((ToggleStateOf $softChk) -eq "Off") "the soft-target switch defaults to OFF" (ToggleStateOf $softChk)
    }
    if ($bcCmb -eq $null) { throw "no BC teacher combo" }
    $items = ComboItemNames $bcCmb
    Write-Host ("  info  bcTeacherCombo items: " + ($items -join " | "))
    Check ($items.Count -ge 2) "the dropdown lists a 'off' entry plus Alpha-Beta levels"
    Check (($items -join " ") -match "off") "the first entry is 'off' (default: no cloning)"

    # ---- 3. negative control: with NO BC-capable student the dropdown must stay disabled ----
    # The negative control had to change with the semantics: it used to be "no AB opponent",
    # which is no longer a reason to disable anything (the teacher is in the dropdown now).
    # The real blocker is "nobody on the board has a policy head" = pure search vs pure search.
    #
    # The human-game AI ("agentComboBox") is part of the enable rule too (in a human game it
    # IS the student), so pin it to a pure-search agent first -- otherwise this control would
    # depend on whatever the app defaults to, and a BC-capable default would make the step
    # fail while the feature is fine.
    $null = Select-ComboItemByMatch $root "agentComboBox" $PureSearchMatch
    $null = Select-ComboItemByMatch $root "matchAComboBox" $NoStudentAMatch
    $null = Select-ComboItemByMatch $root "matchBComboBox" $NoStudentBMatch
    Start-Sleep -Milliseconds 600
    Check (-not (Enabled (FindByObjectName $root "bcTeacherCombo"))) `
        "dropdown disabled when neither side can be a student" `
        ($NoStudentAMatch + " vs " + $NoStudentBMatch)

    # ---- 4. the real pair: student + a NON-Alpha-Beta opponent ----
    # This is the core of the 2026-10 user requirement: the opponent is MCTS, i.e. there is
    # no AB anywhere on the board, and the teacher still comes from the dropdown.
    $selA = Select-ComboItemByMatch $root "matchAComboBox" $StudentMatch
    $selB = Select-ComboItemByMatch $root "matchBComboBox" $OpponentMatch
    Check ($selA -ne "") "selected the student (A)" $selA
    Check ($selB -ne "") "selected a NON-Alpha-Beta opponent (B)" $selB
    Start-Sleep -Milliseconds 600
    $bcCmb = FindByObjectName $root "bcTeacherCombo"
    Check (Enabled $bcCmb) "dropdown enabled with a student + a non-AB opponent (teacher is chosen, not inherited)"

    # ---- 5. pick Alpha-Beta depth 1 in the dropdown, one game only ----
    $picked = Select-ComboItemByMatch $root "bcTeacherCombo" $TeacherItemMatch
    Check ($picked -ne "") "selected the teacher in the dropdown" $picked
    Start-Sleep -Milliseconds 400
    # ---- 5a. [2026-10] turn the soft-target switch ON for this run ----
    # Why the ON state is the one exercised here (and the default stays OFF): the OFF path is
    # the historical one, covered by every earlier run, while the soft path is the new code --
    # and it is the one whose READINGS can be misread (its CE has a floor of H(t), so the
    # report must carry targetH). Turning it on also checks that the switch is reachable and
    # that its enable rule follows the dropdown (both are OFF/disabled when there is no
    # student, see step 3's negative control).
    $softChk = FindByObjectName $root "bcSoftCheck"
    if ($softChk -ne $null) {
        Check (Enabled $softChk) "soft-target switch enabled together with the teacher dropdown"
        if ((ToggleStateOf $softChk) -ne "On") {
            try { $softChk.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Toggle() } catch { }
            Start-Sleep -Milliseconds 300
        }
        Check ((ToggleStateOf $softChk) -eq "On") "soft-target switch really turned ON" (ToggleStateOf $softChk)
    }
    Check (Set-Spin $root "gamesSpin" 1) "gamesSpin set to 1"

    # ---- 5b. "探索+预训练" (per-move rollout pretraining) ----
    # [2026-10 user question] "勾选行为克隆训练后能否勾选 rollout 预训练" -- the two controls
    # are independent: nothing in the BC path touches preTrainCheck (verified in the source),
    # and the BC dropdown's own enable rule never looks at it. So the DEFAULT path here only
    # switches pretraining OFF to keep the run short (pretraining makes every move several
    # times slower). `-KeepPreTrain` is the run that actually exercises the combination: it
    # asserts the box is still enabled + ON **while a BC teacher is selected** (that is the
    # literal answer) and then lets the match run with both on.
    $pre = FindByObjectName $root "preTrainCheck"
    if ($KeepPreTrain) {
        if ((ToggleStateOf $pre) -ne "On") {
            try { $pre.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Toggle() } catch { }
            Start-Sleep -Milliseconds 300
        }
        Check (Enabled $pre) "pretraining checkbox still ENABLED while a BC teacher is selected"
        Check ((ToggleStateOf (FindByObjectName $root "preTrainCheck")) -eq "On") `
            "pretraining checkbox is ON together with the BC teacher (both can be on)"
    } else {
        if ((ToggleStateOf $pre) -eq "On") {
            try { $pre.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Toggle() } catch { }
            Start-Sleep -Milliseconds 200
        }
        Check ((ToggleStateOf (FindByObjectName $root "preTrainCheck")) -eq "Off") `
            "per-move pretraining switched off (keeps this check short; BC does not need it)"
    }

    # ---- 6. start the match ----
    $playBtn = FindByObjectName $root "selfPlayBtn"
    Check ($playBtn -ne $null) "selfPlayBtn found"
    try {
        $playBtn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
        Check $true "selfPlayBtn invoked (match started)"
    } catch {
        Check $false "selfPlayBtn invoked" $_.Exception.Message
    }

    # ---- 7. while the match runs the dropdown must be locked ----
    Start-Sleep -Seconds 3
    Check (-not (Enabled (FindByObjectName $root "bcTeacherCombo"))) `
        "dropdown locked while the match is running (the setting is per-match)"

    # ---- 8. the FIRST live BC line proves the mechanism ----
    # Why this is the primary assertion instead of the per-match summary: the summary only
    # exists once the game is over, and a full game can be long (300 plies x ~0.5 s). The
    # live line appears after the student's **first** move, so it pins the same mechanism
    # (student move -> sample -> actor update) without waiting for the game to end.
    # The line carries ASCII markers ("[BC] ... samples= updates= CE=") for exactly this.
    $text = ""
    $liveLine = ""
    $dlLive = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $dlLive) {
        Start-Sleep -Milliseconds 700
        if ($proc.HasExited) { break }
        $t = EditTextContaining $root "[BC]"
        if ($t -eq "") { continue }
        $m = [regex]::Match($t, "\[BC\][^\r\n]*samples=[0-9]+[^\r\n]*")
        if ($m.Success) {
            $liveLine = $m.Value
            $text = $t
            break
        }
    }
    Check ($liveLine -ne "") "the in-match BC produced samples + updates (live line)" $liveLine
    if ($liveLine -ne "") {
        $s = [regex]::Match($liveLine, "samples=([0-9]+)")
        $u = [regex]::Match($liveLine, "updates=([0-9]+)")
        Check ($s.Success -and [int]$s.Groups[1].Value -gt 0 -and
               $u.Success -and [int]$u.Groups[1].Value -gt 0) `
            "samples>0 and updates>0" $liveLine
        Check ($liveLine -match "CE=[0-9]") "the line carries a CE reading" $liveLine
    }
    # ---- 8b. the live line names the dropdown-selected teacher (and NOT the opponent) ----
    # The board's student is PPO+MCTS (MLP) and its opponent is MCTS -- i.e. the teacher shown
    # here can only have come from the dropdown. This is the assertion that pins the new
    # semantics, and it is made on the LIVE line on purpose: during a match the panel only
    # APPENDS these lines (the whole report body is re-rendered at panel-refresh moments, and
    # this script runs with per-move pretraining OFF, so those moments are rare). Asserting on
    # the report body here was the first version's false FAIL -- the marker was right there in
    # the panel at match end, just not yet during play.
    if ($liveLine -ne "") {
        # NOTE: no closing bracket in the pattern -- the live line now reads
        # `[teacher-depth=N soft=M]`, so requiring `]` right after the digits matched nothing
        # (that was this script's own false FAIL, not a product defect).
        $tm = [regex]::Match($liveLine, "\[teacher-depth=([0-9]+)\b")
        Check ($tm.Success -and [int]$tm.Groups[1].Value -eq $TeacherDepth) `
            ("the live line says the teacher is the dropdown's depth=" + $TeacherDepth + " entry") $liveLine
        # [2026-10] The live line must also say WHICH target shape was in effect: the two shapes
        # are not comparable (soft targets have a CE floor of H(t)), and a switch that silently
        # did nothing would otherwise look identical to "it works, the numbers just moved".
        $sm = [regex]::Match($liveLine, "soft=([0-9]+)")
        Check ($sm.Success -and [int]$sm.Groups[1].Value -eq 1) `
            "the live line reports the soft-target switch as ON (soft=1)" $liveLine
    }

    # ---- 8c. let it reach at least 8 actor updates before stopping ----
    # Why this wait is not optional: the FIDELITY readout (the two "最新/均值" series and the
    # "最近 CE" tail, asserted in step 9b) only starts after kBcFidEvery=4 updates **and** a
    # window of at least kBcFidMin=8 samples. The live progress line is printed at update 1
    # and then every 8 -- so "a line with updates>=8" is exactly the point where the fidelity
    # sample exists. Without this wait the script is a race: one run got 25 updates before the
    # stop (readout present) and the next got 3 (readout "暂无"), and the CE assertions then
    # fail while the feature is fine. That was a real false FAIL of this script.
    $updatesSeen = 0
    if ($liveLine -ne "") {
        $um = [regex]::Match($liveLine, "updates=([0-9]+)")
        if ($um.Success) { $updatesSeen = [int]$um.Groups[1].Value }
    }
    $dlUp = (Get-Date).AddSeconds(240)
    while ($updatesSeen -lt 8 -and (Get-Date) -lt $dlUp) {
        Start-Sleep -Milliseconds 700
        if ($proc.HasExited) { break }
        $t = EditTextContaining $root "[BC]"
        if ($t -eq "") { continue }
        $mm = [regex]::Matches($t, "updates=([0-9]+)")
        if ($mm.Count -gt 0) {
            $last = [int]$mm[$mm.Count - 1].Groups[1].Value
            if ($last -gt $updatesSeen) { $updatesSeen = $last }
        }
        if ($updatesSeen -ge 8) { break }
    }
    Write-Host ("  info  actor updates before the stop: " + $updatesSeen + " (need >= 8 for the fidelity readout)")
    Check ($updatesSeen -ge 8) "the match reached >= 8 actor updates (so the fidelity readout exists)"

    # ---- 8d. the online-path counter is checked after the summary is read (step 9b) ----
    # [2026-10 user question] "勾选行为克隆训练后能否勾选 rollout 预训练" -- the two controls are
    # independent, and "both on" must not mean "one of them silently did nothing". BC's evidence
    # is the live line above; the ONLINE path's evidence is `onlineSteps=` in the summary (it
    # counts how many times the online path reported a loss, so 0 = it did not run). That field
    # exists for this check: the first version tried to read the loss chart's HelpText, and Qt's
    # UIA bridge does not expose it at all (measured: empty string -- which is also why
    # verify_match_ui.ps1's "loss chart = " line has never printed anything).

    # ---- 9. stop the match on purpose, then read the per-match summary ----
    # Stopping it (same button, second press) is not a shortcut for correctness: it exercises
    # the abort path AND makes the summary arrive quickly and deterministically, instead of
    # waiting for a game that may run 300 plies. The abort is checked once per ply
    # (playMatchGame), so the summary lands within a move or two.
    $playBtn = FindByObjectName $root "selfPlayBtn"
    try {
        $playBtn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
        Write-Host "  info  asked the match to stop (second press on the match button)"
    } catch {
        Write-Host ("  info  could not press stop: " + $_.Exception.Message)
    }

    $summaryLine = ""
    $dl2 = (Get-Date).AddSeconds($MatchTimeoutSec)
    while ((Get-Date) -lt $dl2) {
        Start-Sleep -Milliseconds 700
        if ($proc.HasExited) { break }
        $t = EditTextContaining $root "match-summary"
        if ($t -ne "") {
            $text = $t
            # NOTE the pattern: the product writes "本场汇总 (match-summary): samples=...", so
            # the token is followed by ')' and not directly by ':'. Requiring the colon right
            # after the token matched nothing while the text was right there in the panel
            # (the diagnostic run printed "editTextContainingFound=True" and an empty match).
            $summaryLine = [regex]::Match($t, "match-summary[^\r\n]*").Value
            break
        }
    }
    Check ($summaryLine -ne "") "the per-match BC summary reached the panel" $summaryLine
    if ($summaryLine -eq "") {
        # Diagnostic: what does the panel actually hold right now? Without this the only
        # visible symptom is "the summary never arrived", which cannot distinguish
        # "the match never ended" from "the report never reached the panel".
        $t = PanelText $root
        Write-Host ("  info  panel text length: " + $t.Length +
                    "; contains match-summary: " + ($t -match "match-summary") +
                    "; contains [BC]: " + ($t -match "\[BC\]"))
        if ($t.Length -gt 0) {
            $tail = $t.Substring([Math]::Max(0, $t.Length - 1200))
            Write-Host "  ---- panel tail ----"
            foreach ($l in ($tail -split "`r?`n")) { if ($l.Trim() -ne "") { Write-Host ("    " + $l) } }
        }
    }

    if ($summaryLine -ne "") {
        $samples = [regex]::Match($summaryLine, "samples=([0-9]+)")
        $updates = [regex]::Match($summaryLine, "updates=([0-9]+)")
        $missed = [regex]::Match($summaryLine, "targetMissed=([0-9]+)")
        Check ($samples.Success -and [int]$samples.Groups[1].Value -gt 0) `
            "the summary counts samples" ("samples=" + $(if ($samples.Success) { $samples.Groups[1].Value } else { "?" }))
        Check ($updates.Success -and [int]$updates.Groups[1].Value -gt 0) `
            "the summary counts actor updates" ("updates=" + $(if ($updates.Success) { $updates.Groups[1].Value } else { "?" }))
        Check ($missed.Success -and [int]$missed.Groups[1].Value -eq 0) `
            "no sample was lost to a target miss (caliper consistency)" ("targetMissed=" + $(if ($missed.Success) { $missed.Groups[1].Value } else { "?" }))
        # ---- the online path's counter (see step 8d) ----
        $onl = [regex]::Match($summaryLine, "onlineSteps=([0-9]+)")
        $onlN = if ($onl.Success) { [int]$onl.Groups[1].Value } else { -1 }
        Write-Host ("  info  onlineSteps (在线训练/探索+预训练 上报损失的次数): " + $onlN +
                    "   (KeepPreTrain=" + [bool]$KeepPreTrain + ")")
        Check ($onl.Success) "the summary carries the online-training counter (onlineSteps=)" $summaryLine
        if ($KeepPreTrain) {
            Check ($onl.Success -and $onlN -gt 0) `
                "pretraining stayed ON and really trained alongside BC (onlineSteps>0)" $summaryLine
        }
        Check ($text -match "CE[=:]") "the summary carries a CE reading"
        # ---- the soft-target readings (see step 5a) ----
        # Two things must be true when the switch is on: (a) the summary says so (`soft=1`),
        # and (b) the report carries the AVERAGE TARGET ENTROPY H(t). (b) is not cosmetic:
        # with soft targets the loss is CE = H(t) + KL(t||pi), so a CE number without H(t)
        # next to it cannot be interpreted at all ("did the student get worse, or was the
        # teacher itself unsure?"). H(t) must be > 0 -- a soft target that collapses onto one
        # move would mean the multi-depth vote did nothing and we fell back to one-hot.
        # That is exactly why this run selects depth=2: at depth=1 there is a single depth to
        # vote, so the soft target **legitimately** degenerates to one-hot and H(t)=0 (the
        # product says so in the report). Asserting "> 0" only means something above depth 1.
        $softFlag = [regex]::Match($summaryLine, "soft=([0-9]+)")
        Check ($softFlag.Success -and [int]$softFlag.Groups[1].Value -eq 1) `
            "the summary says the soft-target switch was ON (soft=1)" $summaryLine
        $th = [regex]::Match($text, "targetH=([0-9]+\.[0-9]+)")
        if ($th.Success) {
            $thV = [double]$th.Groups[1].Value
            # ⚠ The assertion must follow the CALIPER, not a wish: with teacher depth=1 the
            # multi-depth vote has a single depth to vote, so the soft target **legitimately**
            # degenerates to one-hot and H(t) is exactly 0 (that property is pinned in
            # `test_bc [9]`). Asserting "> 0" there was this script's own false FAIL. The
            # product now explains that case in the report, and this checks the explanation.
            if ($TeacherDepth -le 1) {
                Check ($thV -eq 0.0) `
                    "the report degrades soft targets to one-hot at teacher depth=1 (targetH must be exactly 0)" `
                    ("targetH=" + $thV)
                Check ($text -match "只有一票") `
                    "the report EXPLAINS the depth=1 degeneration instead of printing a bare 0" `
                    "报告里那一行 = 深度=1 时只有一票 ⇒ 目标退化成 one-hot"
            } else {
                Check ($thV -gt 0.0) `
                    ("the report carries a POSITIVE average target entropy H(t) (soft target really is a distribution; teacher depth=" + $TeacherDepth + ")") `
                    ("targetH=" + $thV)
            }
        } else {
            Check $false "the report carries targetH= (the soft-target reading)" "targetH not found in the panel"
        }
        Write-Host "  ---- BC lines of the panel ----"
        # The filter must include the soft-target readings -- otherwise the very lines this run
        # exists to check (targetH / the degeneration warning) are the ones MISSING from the log,
        # and a failure can only be diagnosed by re-running under a debugger (this happened).
        foreach ($l in ($text -split "`r?`n")) {
            if ($l -match "BC|match-summary|CE[=:]|critic|targetH|soft targets|只有一票") {
                Write-Host ("    " + $l)
            }
        }
    }

    # ---- 9b. the BC fidelity curve lives in a tab next to the loss chart ----
    # Assertions here are on the widget tree + the readout line, not on pixels: the curve
    # itself is drawn (UIA cannot read it), but CurveChart puts "最新/均值/样本数" into its
    # accessible name and the label under it carries the CE -- exactly the pair of numbers
    # this feature exists to expose.
    #
    # IMPORTANT: the BC widgets live in tab page 1. Qt's UIA bridge does not expose the
    # widgets of a hidden tab page, so the first version of this check reported
    # "bcChart NOT FOUND" while the feature was fine -- the tab has to be selected first.
    # The tab texts carry ASCII markers ("训练损失 (loss)" / "行为克隆 (BC)") so this
    # script can pick the right tab without reading CJK.
    $tabs = FindByObjectName $root "chartTabs"
    Check ($tabs -ne $null) "the loss chart area is now a tab widget (chartTabs)"
    $tabSel = $false
    if ($tabs -ne $null) {
        $tabCond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::TabItem)
        $tabItems = $tabs.FindAll([System.Windows.Automation.TreeScope]::Descendants, $tabCond)
        Write-Host ("  info  tabs found: " + $tabItems.Count)
        foreach ($t in $tabItems) {
            $n = PropOf $t ([System.Windows.Automation.AutomationElement]::NameProperty)
            Write-Host ("    tab: " + $n)
            if ($n -match "BC") {
                try {
                    $t.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
                    Start-Sleep -Milliseconds 500
                    $tabSel = $true
                } catch {
                    Write-Host ("  info  tab Select() failed: " + $_.Exception.Message)
                }
            }
        }
    }
    Check $tabSel "switched to the BC tab (tab page visibility gates UIA exposure)"

    $bcChart = FindByObjectName $root "bcChart"
    Check ($bcChart -ne $null) "the BC fidelity chart exists (bcChart)"
    if ($bcChart -ne $null) {
        $an = PropOf $bcChart ([System.Windows.Automation.AutomationElement]::NameProperty)
        Write-Host ("  info  bcChart readout: " + $an)
        Check ($an -match "[0-9]") "the BC chart reports numbers (readout/accessible name)" $an
    }
    $bcLabel = FindByObjectName $root "bcValueLabel"
    Check ($bcLabel -ne $null) "the BC readout label exists (bcValueLabel)"
    if ($bcLabel -ne $null) {
        $lt = PropOf $bcLabel ([System.Windows.Automation.AutomationElement]::NameProperty)
        Write-Host ("  info  bcValueLabel: " + $lt)
        Check ($lt -match "CE") "the readout line carries the CE (not on the chart: different unit)" $lt
    }

    # ---- 9a2. the value-evaluation tab (2026-10 user wording: "价值评估曲线") ----
    # Why it is checked here (and not in verify_match_ui.ps1): this script already has the tab
    # machinery AND a running match, so it can assert the tab exists and that its readout is in
    # the honest "no reading yet" state instead of an empty/blank line. Getting a real EV point
    # needs a FINISHED, decisive game with >= 64 sampled plies (see the label text) -- that is
    # what the CLI `bench_diag` is for; here we pin the plumbing and the "no fake zero" wording.
    $valueSel = $false
    if ($tabs -ne $null) {
        foreach ($t in $tabItems) {
            $n = PropOf $t ([System.Windows.Automation.AutomationElement]::NameProperty)
            if ($n -match "value") {
                try {
                    $t.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
                    Start-Sleep -Milliseconds 500
                    $valueSel = $true
                } catch { }
            }
        }
    }
    Check $valueSel "switched to the value tab (tab page visibility gates UIA exposure)"
    $valueChart = FindByObjectName $root "valueChart"
    Check ($valueChart -ne $null) "the value-evaluation chart exists (valueChart)"
    if ($valueChart -ne $null) {
        $an = PropOf $valueChart ([System.Windows.Automation.AutomationElement]::NameProperty)
        Write-Host ("  info  valueChart readout: " + $an)
        # accessibleName = title + " · n=<points>"; the title carries the caliper (EV / 0 / 1)
        Check ($an -match "EV") "the value chart title names the caliper (EV)" $an
    }
    $valueLabel = FindByObjectName $root "valueValueLabel"
    Check ($valueLabel -ne $null) "the value readout label exists (valueValueLabel)"
    if ($valueLabel -ne $null) {
        $vt = PropOf $valueLabel ([System.Windows.Automation.AutomationElement]::NameProperty)
        Write-Host ("  info  valueValueLabel: " + $vt)
        Check ($vt -ne "") "the value readout line is not blank (it explains the state)" $vt
    }
    # ⚠ 必须切回 BC 那个 tab 再往下走: 紧接着的 9b 是"双击 BC 曲线 -> 放大窗口", 而 Qt 的
    # UIA 桥**不暴露隐藏 tab 页里的控件** —— 刚才为了断言 valueChart 切到了价值页, 于是
    # bcChart 从 UIA 树里消失了, 双击自然打在空气上 (本轮实测: 三条与双击有关的断言全红,
    # 而功能是好的)。这与"隐藏 tab 页里的控件不存在"是同一条纪律, 只是这次是**自己把
    # 自己切走了**。
    if ($tabs -ne $null) {
        foreach ($t in $tabItems) {
            $n = PropOf $t ([System.Windows.Automation.AutomationElement]::NameProperty)
            if ($n -match "BC") {
                try {
                    $t.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
                } catch { }
                Start-Sleep -Milliseconds 500
            }
        }
    }

    # ---- 9b. double-click the BC curve -> the enlarged window (2026-10 user report) ----
    # The report was literally "double-clicking the behaviour curve does nothing". Cause:
    # the handler was connected in setupMetricsPanel(), which runs BEFORE setupChartTabs()
    # creates m_bcChart, so `if (m_bcChart != nullptr)` silently skipped it -- no compile
    # error, no runtime warning, just a dead double-click. UIA has no double-click pattern
    # (only Invoke), so this drives real mouse input via mouse_event, exactly like the
    # loss-chart check in verify_match_ui.ps1.
    #
    # Two things are asserted, and the second is the interesting one:
    #   1. a top-level window titled "行为克隆保真度 (放大)" appears;
    #   2. its readout line is the SAME text as the source label (modulo the "读数" /
    #      "克隆保真度" prefix) -- including the CE tail, which is NOT a series on the
    #      chart and therefore has to be handed to the dialog explicitly. A window that
    #      opens but shows fewer numbers would read as "CE disappeared".
    $bcDlgName = "行为克隆保真度 (放大)"
    $bcDlg = $null
    $bcDlgText = ""
    if ($bcChart -ne $null) {
        $r = $bcChart.Current.BoundingRectangle
        if ($r.Width -gt 4 -and $r.Height -gt 4) {
            [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point(
                ([int]($r.X + $r.Width / 2)), ([int]($r.Y + $r.Height / 2)))
            Start-Sleep -Milliseconds 200
            Add-Type -MemberDefinition '[DllImport("user32.dll")]public static extern void mouse_event(int f,int x,int y,int d,int e);' -Name UbcDbl -Namespace WbcDbl -PassThru | Out-Null
            for ($c = 0; $c -lt 2; $c++) {
                [WbcDbl.UbcDbl]::mouse_event(0x0002, 0, 0, 0, 0)
                Start-Sleep -Milliseconds 40
                [WbcDbl.UbcDbl]::mouse_event(0x0004, 0, 0, 0, 0)
                Start-Sleep -Milliseconds 60
            }
            Start-Sleep -Milliseconds 800
        }
        # The dialog is a QDialog(parent=main window) -> Windows treats it as an OWNED
        # top-level window, which UIA hangs under the main window rather than under the
        # desktop root's Children (measured in verify_match_ui.ps1: Children=False,
        # Descendants=True). So search the desktop root's Descendants (a UIA-side search,
        # NOT a local tree walk: the desktop has hundreds of elements and a walk here
        # costs seconds per call).
        $dlgCond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::NameProperty, $bcDlgName)
        $bcDlg = [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants, $dlgCond)
        if ($bcDlg -ne $null) {
            foreach ($e in (AllDescendants $bcDlg)) {
                $n = PropOf $e ([System.Windows.Automation.AutomationElement]::NameProperty)
                if ($n -like "读数*") { $bcDlgText = $n; break }
            }
        }
    }
    Check ($bcDlg -ne $null) ("double-click on the BC curve opens the enlarged window (" + $bcDlgName + ")")
    if ($bcDlgText -ne "") { Write-Host ("  info  enlarged readout = " + $bcDlgText) }
    $mirror = $false
    if ($bcDlgText -ne "" -and $lt -ne "") {
        # Strip the differing prefixes ("读数" vs "克隆保真度"), then require the rest to be
        # byte-identical: that checks "the window shows the same numbers", not merely
        # "a window opened".
        $a = $bcDlgText -replace "^读数\s*", ""
        $b = $lt -replace "^克隆保真度\s*", ""
        $mirror = ($a -eq $b) -and ($a -ne "")
        Write-Host ("  info  mirror vs source label = {0}  (strip the prefix, compare the rest)" -f $mirror)
    }
    Check $mirror "the enlarged window mirrors the source readout (incl. the CE tail)"
    Check ($bcDlgText -match "CE") "the enlarged window carries the CE (it is not a chart series)"
    if ($bcDlg -ne $null) {
        # close it through WindowPattern (focus independent) and assert it is gone -- a
        # leftover window would sit on top of the app for the next check.
        $wp = $null
        if ($bcDlg.TryGetCurrentPattern([System.Windows.Automation.WindowPattern]::Pattern, [ref]$wp)) {
            $wp.Close()
        }
        Start-Sleep -Milliseconds 600
        $stillThere = [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants, $dlgCond)
        Check ($stillThere -eq $null) "the enlarged window closes again"
    }

    # ---- 10. the dropdown comes back after the match ----
    $back = $false
    $dl3 = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $dl3) {
        Start-Sleep -Milliseconds 500
        if (Enabled (FindByObjectName $root "bcTeacherCombo")) { $back = $true; break }
    }
    Check $back "dropdown unlocked again after the match"

if ($SkipHuman) {
    # [2026-10 user 口径] "不进行人机对弈测试" —— 这一段会**真的在屏幕上下一局棋**
    # (盲走红兵, 用户会看着自己被将军/将死), 所以给它一个开关。跳过时**不产生任何断言**:
    # 它不影响其它段的结论, 只是少覆盖一段 (那段自己的结论保持"未测", 不被当成通过)。
    Write-Host "  info  跳过 human game 段 (-SkipHuman): 这一段会真的在人机上下一局棋"
} else {
        # ---- 10b. HUMAN-vs-AI: the BC curve must be armed AND must really draw points ----
        # Why this section exists (2026-10 user report): in a human game the BC chart was **never
        # armed** -- the series are created on `matchStarted`, which only fires for Agent-vs-Agent,
        # and `CurveChart::addPoint` **silently returns** when the series does not exist. So BC was
        # training (the panel showed samples=/updates=) while the curve stayed empty forever.
        #
        # Two separate claims are checked, because "blank chart" has two different causes:
        #   (a) not armed at all          -> the marker line "[BC] 曲线已建线 (chart-armed) ... [mode=human]"
        #       (product text, printed by MainWindow::armBcChartSeries so a script can tell the two apart);
        #   (b) armed but too few samples -> the fidelity readout needs >= 8 samples AND every 4 actor
        #       updates, so the first point only appears after the 8th AI move.
        # (b) is therefore measured for real: this script PLAYS the human side (it has to -- the AI
        # only moves after the human does) until the chart reports a point. The chart's point count is
        # readable because CurveChart puts it into its accessible **Name** ("... · n=<points>");
        # Description would have been the natural place, but Qt's UIA bridge does not expose it.
        $aiSide = Select-ComboItemByMatch $root "agentComboBox" $StudentMatch
        Check ($aiSide -ne "") "human game: the AI side is a BC-capable student" $aiSide
        $null = Select-ComboItemByMatch $root "bcTeacherCombo" $TeacherItemMatch
        Start-Sleep -Milliseconds 400
        # start a fresh human game (also clears whatever the match above left on the board)
        $resetBtn = FindByObjectName $root "resetBtn"
        if ($resetBtn -ne $null) {
            try { $resetBtn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke() } catch { }
            Start-Sleep -Milliseconds 800
        }
        $boardEl = FindByObjectName $root "gameWidget"
        if ($boardEl -eq $null) {
            Check $false "human game: the board widget (gameWidget) was found for clicking"
        } else {
            $br = $boardEl.Current.BoundingRectangle
            Write-Host ("  info  board rect = " + $br)
            # ChessBoard geometry: offsetX/offsetY = 50, gridSize = 60 (chessboard.h);
            # Board-Point(col,row) below is the same mapping verify_human_vs_ai.ps1 uses.
            $script:bcBoardLeft = [int]$br.X
            $script:bcBoardTop = [int]$br.Y
            # Candidate human moves: push a red pawn straight ahead, centre files first, from the
            # back rank towards the enemy. Ordered so that "keep pushing the same pawn" comes first;
            # an illegal candidate simply costs one wait (see the loop) instead of aborting.
            $cand = New-Object System.Collections.ArrayList
            foreach ($col in @(4, 2, 6, 0, 8)) {
                for ($row = 6; $row -ge 1; $row--) {
                    # ⚠ 括号不能省: PowerShell 里逗号比算术**绑得更紧**, 写成
                    # `@($col, $row, $col, $row - 1)` 会被解析成 `(数组) - 1` ⇒ 运行期报
                    # "does not contain a method named 'op_Subtraction'" (本轮实测踩到)。
                    [void]$cand.Add(@($col, $row, $col, ($row - 1)))
                }
            }
            # ---- this loop deliberately does NOT try to detect a single move (three versions) ----
            # v1 compared the LAST "samples=N" in the panel with the max seen so far -- the panel
            #    still held the match's lines, so the first candidate "succeeded" instantly against a
            #    stale line and the bookkeeping was fiction (2 "moves" while ~12 had really happened).
            # v2 counted "[BC] ... samples=" lines and waited for the count to grow -- honest, but the
            #    product prints that line only at `updates == 1` and every `kBcReportEveryUpdates = 8`
            #    updates (a deliberate "don't spam the panel" choice), so moves 2..7 produce **no new
            #    line** and were all mis-read as illegal (the run then stopped after one move).
            # v3 (this one): click a candidate, wait a SHORT settled time, repeat; then make every
            #    claim on the panel text **sliced at the last arm marker** (everything after it belongs
            #    to the current human game). That slice is what makes the claims exact:
            #      * the arm marker line itself -> the curve was armed in human mode (the fix);
            #      * a human-game "samples=" line -> BC really ran in this game;
            #      * a human-game "samples=8" line -> >= 8 actor updates, so a fidelity point MUST
            #        exist (<= that is exactly the boundary: 8 updates x window >= 8 samples).
            #    A terminal state is handled explicitly: blind pawn pushes lose to the AI, and the
            #    readings are per-GAME in human mode, so the script closes the dialog and opens a new
            #    game instead of pretending the old one can still reach 8 moves.
            $deadline = (Get-Date).AddSeconds($HumanPhaseSec)
            $tries = 0
            $games = 0
            $points = 0
            # ⚠ 必须**显式初始化**为 "": 没初始化时它是 $null, 而 PowerShell 里 `$null -ne ""`
            # 是 **True** —— 于是"arm marker 到了吗"那条断言会假通过, 而后面两条真正检查内容的
            # 断言则失败 (本轮实测: 3 条断言给出互相矛盾的结果, 根因就是这一行)。
            $armLine = ""
            foreach ($mv in $cand) {
                if ((Get-Date) -gt $deadline) { break }
                $endDlg = Find-BcEndDialog
                if ($endDlg -ne $null) {
                    Write-Host ("  info  human game: terminal dialog appeared (" +
                                $endDlg.Current.Name + ") -> closing it and starting a new game")
                    try {
                        $wp = $null
                        if ($endDlg.TryGetCurrentPattern(
                                [System.Windows.Automation.WindowPattern]::Pattern, [ref]$wp)) {
                            $wp.Close()
                        }
                    } catch { }
                    Start-Sleep -Milliseconds 500
                    $rb = FindByObjectName $root "resetBtn"
                    if ($rb -ne $null) {
                        try { $rb.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke() } catch { }
                    }
                    Start-Sleep -Milliseconds 900
                    $games++
                    continue
                }
                $tries++
                $p1 = BcBoardPoint $mv[0] $mv[1]
                BcClick $p1[0] $p1[1]
                Start-Sleep -Milliseconds 120
                $p2 = BcBoardPoint $mv[2] $mv[3]
                BcClick $p2[0] $p2[1]
                Start-Sleep -Milliseconds $MoveSettleMs
                # 顺手把 arm marker 记下来: 面板在**刷新时机**会被整段重写(setPlainText), 附加的
                # 这一行随后就没了 —— 所以不能只在最后去读它, 要在它还活着的时候抓住。
                if ($armLine -eq "") {
                    $am = [regex]::Match((EditTextContaining $root "[BC]"),
                                         "\[BC\][^\r\n]*chart-armed[^\r\n]*\[mode=human\][^\r\n]*")
                    if ($am.Success) { $armLine = $am.Value }
                }
                $ch = FindByObjectName $root "bcChart"
                if ($ch -ne $null) {
                    $nm = PropOf $ch ([System.Windows.Automation.AutomationElement]::NameProperty)
                    $pm = [regex]::Match($nm, "n=([0-9]+)")
                    if ($pm.Success) { $points = [int]$pm.Groups[1].Value }
                }
                if ($points -gt 0) { break }
            }
            # ---- 断言基于"最后一个 arm marker 之后的面板切片" ----
            # ⚠ 切点必须用**整行匹配的 Index**, 不能用 LastIndexOf("[mode=human]") —— 后者会从
            # marker 行**内部**切开, 于是 "chart-armed" 那半截被切掉, 提取出来永远是空串
            # (本轮实测: 5 条断言里 3 条因此假失败, 而功能是好的)。
            $panel = EditTextContaining $root "[BC]"
            $armMatches = [regex]::Matches(
                $panel, "\[BC\][^\r\n]*chart-armed[^\r\n]*\[mode=human\][^\r\n]*")
            $cut = -1
            if ($armMatches.Count -gt 0) {
                $lastArm = $armMatches[$armMatches.Count - 1]
                if ($armLine -eq "") { $armLine = $lastArm.Value }
                $cut = $lastArm.Index
            }
            $humanPart = if ($cut -ge 0) { $panel.Substring($cut) } else { "" }
            $hSamples = [regex]::Matches($humanPart, "\[BC\][^\r\n]*samples=([0-9]+)")
            $hMax = -1
            foreach ($m in $hSamples) {
                $v = [int]$m.Groups[1].Value
                if ($v -gt $hMax) { $hMax = $v }
            }
            # ---- 断言 ----
            # 两条**确定性**断言 + 一条**条件**断言。为什么做成条件: 保真度第一个点要 >= 8 次
            # actor 更新 (kBcFidEvery=4 x 窗口下界 kBcFidMin=8), 而脚本是**盲走红兵**的 ——
            # 用户在自己屏幕上看到的那两幕("被将军了"、"黑方已经胜利")就是这个: AI 把不会看局面的
            # 红方将死, 一局常常走不到 8 手。而人机那条路的曲线是**每局清零**的, 所以"这一局没到 8
            # 手"时曲线本来就该是空的 —— 那是口径, 不是缺陷。
            #   确定性 (1): 这一局建了线 (arm marker, mode=human) —— 这正是本次修的那个缺口;
            #   确定性 (2): 这一局 BC 真的跑过 (有 samples= 行);
            #   条件   (3): 面板出现 samples=8 (>= 8 次更新) 时, 曲线**必须**已经有点。
            #              上一轮的运行里这条真的触发过 (实测 chart n=1)。
            Write-Host ("  info  human game: " + $tries + " candidate moves clicked, " + $games +
                        " terminal dialog(s) -> new game")
            Write-Host ("  info  arm marker: " + $armLine)
            Write-Host ("  info  human game: live BC lines in THIS game = " + $hSamples.Count +
                        " (max samples=" + $hMax + "), chart points = " + $points)
            # ⚠ 判"空"要用 IsNullOrEmpty: `$x -ne ""` 对 $null 为 **True** (PowerShell 的经典坑)
            Check (-not [string]::IsNullOrEmpty($armLine)) `
                "human game: the arm marker arrived ([chart-armed] … [mode=human])"
            Check ($armLine -match "chart-armed") `
                "human game: the BC curve got armed (series created) [chart-armed]" $armLine
            Check ($armLine -match "\[teacher-depth=$TeacherDepth\]") ("human game: armed with the dropdown teacher (depth=" + $TeacherDepth + ")") $armLine
            Check ($hSamples.Count -ge 1) "human game: BC produced a live line in this game (samples=)"
            if ($hMax -ge 8) {
                Check ($points -gt 0) ("human game: >= 8 updates, so the BC curve drew points (chart n=" +
                                       $points + ")")
            } else {
                Write-Host ("  info  跳过'曲线出点'那条断言: 这一局只走到 samples=" + $hMax +
                            " (被 AI 将死 / 预算用尽; 保真度第一个点要 >= 8), 曲线到那一刻本来就不该有点")
            }
        }
}

    # ---- 11. NOTE: no "weights unchanged" assertion here ----
    # Deliberately dropped from the first version of this script, which failed on it: an
    # Agent-vs-Agent match **saves the participants' weights to the standard paths at the
    # end** -- that is the app's documented per-match behaviour and it has its own script
    # (verify_chess_saves_weights.ps1). So "weights/ changed" after a match says nothing
    # about BC. The claim this file can make is the narrower one: BC itself never writes
    # (the in-match path only calls the actor update; the report says so, and the CLI's
    # `--save` is an explicit opt-in).
    $weightsAfter = StandardWeightSnapshot $exeDir
    Write-Host ("  info  standard weight files before/after: " + $weightsBefore.Count +
                "/" + $weightsAfter.Count + " (the per-match save is expected, see above)")
} finally {
    # ---- shutdown: kill the app, and verify it is really gone ----
    # Two reasons this must be a kill and must be verified:
    #   1. closing the window would run the app's "save every instantiated agent on exit",
    #      overwriting the user's weights with whatever this script trained;
    #   2. a leftover chess.exe is worse than a failed check -- it keeps the GUI on screen
    #      (the user can then interact with it, and the next run has two instances), and the
    #      first version of this script could leave exactly that behind if the kill threw.
    if ($proc -ne $null -and -not $proc.HasExited) {
        try { $proc.Kill() } catch { }
        Start-Sleep -Seconds 2
        try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
        Start-Sleep -Milliseconds 500
    }
    $left = @(Get-Process -Name "chess" -ErrorAction SilentlyContinue)
    if ($left.Count -gt 0) {
        Write-Host ("  info  closing " + $left.Count + " leftover chess.exe process(es)...")
        foreach ($p in $left) { try { Stop-Process -Id $p.Id -Force } catch { } }
        Start-Sleep -Milliseconds 500
        $left = @(Get-Process -Name "chess" -ErrorAction SilentlyContinue)
    }
    Check ($left.Count -eq 0) "chess.exe stopped after the check (no leftover window)"
}

Write-Host ""
if ($script:fail -eq 0) {
    Write-Host "RESULT: PASS" -ForegroundColor Green
    exit 0
} else {
    Write-Host ("RESULT: FAIL (" + $script:fail + " check(s))") -ForegroundColor Red
    exit 1
}
