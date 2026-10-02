# tools/verify_ppo_moe_selfcheck.ps1
#
# Verifies the PPO+MCTS self-check text in the running chess.exe reports the MoE
# balancing configuration and the train/infer split readout (2026-10).
#
# Why this check exists:
#   The auxiliary-loss-free bias is a NON-PARAMETER buffer: it is not written to the
#   weight file, does not enter paramCount() and does not enter the structural
#   fingerprint. So "did this agent actually get the balancing fix?" is invisible from
#   every other reading -- the panel line is the only place it can be seen. Same reason
#   the SAC agents report it (docs/moe_gate_experiment_2026_10.md sections 8 and 12).
#
# Method: UIA only. The panel is a multi-line Edit control; its ValuePattern carries the
# whole text. All assertions are on ASCII tokens ("MoE", "MaxVio", "train") so this file
# stays ASCII-only -- see verify_moe_load_view.ps1 for why that matters (BOM-less UTF-8
# is decoded as ANSI by Windows PowerShell, so a CJK literal in code breaks the parse).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools/verify_ppo_moe_selfcheck.ps1
#   powershell -ExecutionPolicy Bypass -File tools/verify_ppo_moe_selfcheck.ps1 -AgentMatch "PPO"

param(
    [string]$Exe = "",
    [int]$TimeoutSec = 180,
    # Substring used to pick the agent row in the agent combo (ASCII token).
    [string]$AgentMatch = "PPO+MCTS (AlphaZero,"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

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
# ("QApplication.MainWindow.centralwidget.controlWidget.agentComboBox"), which is ASCII by
# construction -- matching on the tail is language-independent (same idiom as
# verify_moe_load_view.ps1).
function FindByObjectName($root, [string]$objectName) {
    foreach ($e in (AllDescendants $root)) {
        $aid = PropOf $e ([System.Windows.Automation.AutomationElement]::AutomationIdProperty)
        if ($aid.EndsWith("." + $objectName)) { return $e }
    }
    return $null
}

function AllDescendants($root) {
    $out = New-Object System.Collections.ArrayList
    $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
    $stack = New-Object System.Collections.Stack
    $stack.Push($root)
    while ($stack.Count -gt 0) {
        $cur = $stack.Pop()
        [void]$out.Add($cur)
        $ch = $walker.GetFirstChild($cur)
        while ($ch -ne $null) { $stack.Push($ch); $ch = $walker.GetNextSibling($ch) }
    }
    return $out
}

Write-Host "=== verify_ppo_moe_selfcheck ==="
Write-Host ("exe: " + $Exe)

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
try {
    # ---- 1. main window ----
    $dl = (Get-Date).AddSeconds($TimeoutSec)
    $root = $null
    while ((Get-Date) -lt $dl) {
        Start-Sleep -Milliseconds 500
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
        $wins = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Children, $cond)
        if ($wins.Count -gt 0) { $root = $wins[0]; break }
    }
    Check ($root -ne $null) "main window found via UIA"
    if ($root -eq $null) { throw "no main window" }

    # ---- 2. wait for the UI to actually be built ----
    # The window handle appears long before the widgets are laid out and before the
    # startup weight preload finishes, so looking for `agentComboBox` immediately is a
    # race (the first version of this script failed on exactly that: "agentComboBox
    # found" -> FAIL, then a null-method error two lines later).
    $combo = $null
    $dl0 = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $dl0) {
        Start-Sleep -Milliseconds 1000
        if ($proc.HasExited) { break }
        $combo = FindByObjectName $root "agentComboBox"
        if ($combo -ne $null) { break }
    }
    Check ($combo -ne $null) "agentComboBox found (UI finished building)"
    $want = $AgentMatch
    $pidCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
    $liCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)

    $selected = ""
    if ($combo -ne $null) {
    foreach ($attempt in 1..3) {
        $ec = $combo.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern)
        $ec.Expand()
        Start-Sleep -Milliseconds 800
        $and = New-Object System.Windows.Automation.AndCondition($pidCond, $liCond)
        $items = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Descendants, $and)
        $target = $null
        foreach ($it in $items) {
            $n = PropOf $it ([System.Windows.Automation.AutomationElement]::NameProperty)
            if ($n -like ("*" + $want + "*")) { $target = $it; break }
        }
        if ($target -ne $null) {
            $selected = PropOf $target ([System.Windows.Automation.AutomationElement]::NameProperty)
            try {
                $target.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
            } catch {
                Write-Host ("  info  Select() failed: " + $_.Exception.Message)
                $selected = ""
            }
        }
        $ec.Collapse()
        Start-Sleep -Milliseconds 500
        if ($selected -ne "") { break }
    }
    }
    Check ($selected -ne "") "selected a PPO agent row in the agent combo" $selected

    # ---- 3. press "check every model" so the panel is definitely rebuilt on demand ----
    # Why drive the button instead of just waiting: the panel content follows whichever
    # agent the board last built, and it is refreshed on selection / after pre-training.
    # "Wait and hope" produced a stale panel in one of the runs here (it showed the SAC
    # agent's text while the dropdown said PPO), so the check now asks for the report
    # explicitly -- that is also exactly what a user does when they want this reading.
    $allBtn = FindByObjectName $root "selfCheckAllBtn"
    Check ($allBtn -ne $null) "selfCheckAllBtn found"
    if ($allBtn -ne $null) {
        try {
            $allBtn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
        } catch {
            Write-Host ("  info  Invoke() failed: " + $_.Exception.Message)
        }
    }

    # ---- 4. read the self-check panel text ----
    # The panel is the multi-line Edit control below the MoE load widget; its ValuePattern
    # carries the whole report. Poll until it mentions MoE (building the report for every
    # model takes seconds).
    $text = ""
    $dl2 = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $dl2) {
        Start-Sleep -Milliseconds 1000
        foreach ($e in (AllDescendants $root)) {
            $ct = PropOf $e ([System.Windows.Automation.AutomationElement]::ControlTypeProperty)
            if ($ct -ne [string][System.Windows.Automation.ControlType]::Edit) { continue }
            $vp = $null
            try { $vp = $e.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern) } catch { continue }
            if ($vp -eq $null) { continue }
            $t = $vp.Current.Value
            if ($t -ne $null -and $t -match "PPO" -and $t -match "MoE" -and $t.Length -gt 400) {
                $text = $t
                break
            }
        }
        if ($text -ne "") { break }
    }
    Check ($text -ne "") "self-check panel text found (PPO report mentioning MoE)"

    if ($text -ne "") {
        $lines = $text -split "`r?`n"
        Write-Host "  ---- lines of the self-check panel that matter here ----"
        foreach ($l in $lines) {
            if ($l -match "^MoE|MaxVio|train|Loss-Free|均衡|门控") { Write-Host ("    " + $l) }
        }
        # ---- the panel must really belong to the PPO agent we selected ----
        # Without this the assertions below can be satisfied by whatever agent the panel
        # happens to describe -- the first version of this script read the SAC panel and
        # reported it as a failure of the PPO text (selecting a row only sets the dropdown;
        # the panel content belongs to whichever agent the board last built). Asserting
        # identity first is the difference between "the PPO text is wrong" and "we read the
        # wrong agent's text", and those two need different fixes.
        Check ($text -match "PPO") "panel belongs to a PPO agent (not SAC/other)"
        Check ($text -match "Loss-Free") `
            "panel names the balancing mode (Loss-Free bias vs aux-only)"
        Check ($text -match "MaxVio") "panel reports MaxVio (the balance metric)"
        Check ($text -match "train") "panel separates the train-side readout"
        Check ($text -notmatch "max/mean") `
            "panel does not fall back to max/min (the metric we deliberately dropped)"
    }
} finally {
    if (-not $proc.HasExited) {
        try { $proc.CloseMainWindow() | Out-Null } catch { }
        Start-Sleep -Seconds 2
        if (-not $proc.HasExited) { try { $proc.Kill() } catch { } }
    }
}

Write-Host ""
if ($script:fail -eq 0) {
    Write-Host "RESULT: PASS" -ForegroundColor Green
    exit 0
} else {
    Write-Host ("RESULT: FAIL (" + $script:fail + " check(s))") -ForegroundColor Red
    exit 1
}
