# tools/verify_moe_gui_ppo_full.ps1
#
# One session, both readings, for the PPO+MCTS (TB experts) agent:
#   1. selects the agent in the agent combo,
#   2. waits until the self-check panel really describes THAT agent (it must mention
#      "PPO" and NOT "MLP"), pressing "self check all models" so the report is on demand,
#   3. prints the MoE lines of the panel,
#   4. prints the MoE load widget's accessible summary (the machine-readable one).
#
# Why a separate script instead of reusing the two -SelectMoe checks one after another:
#   the widget is fed by the same self-check worker, and the first select after startup
#   can arrive while the agent's net is still being built (the TB backbone now costs
#   ~17.6 s of construction + 1,057 MB of weight reads). A check that polls for ~2 minutes
#   is fine; a check that reads once right after the selection is a flake generator.
#   This script polls until the panel text is the right agent's, then reads the widget.
#
# ASCII only, no BOM.

param(
    [string]$Exe = "",
    [int]$TimeoutSec = 300
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

function PropOf($el, $prop) {
    try {
        $v = $el.GetCurrentPropertyValue($prop)
        if ($v -eq [System.Windows.Automation.AutomationElement]::NotSupported) { return "" }
        return [string]$v
    } catch { return "" }
}

function Descendants($root) {
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

function FindByObjectName($root, [string]$objectName) {
    foreach ($e in (Descendants $root)) {
        $aid = PropOf $e ([System.Windows.Automation.AutomationElement]::AutomationIdProperty)
        if ($aid.EndsWith("." + $objectName)) { return $e }
    }
    return $null
}

function MoeWidget($root) {
    foreach ($e in (Descendants $root)) {
        if ((PropOf $e ([System.Windows.Automation.AutomationElement]::ClassNameProperty)) -eq "MoeLoadView") {
            return $e
        }
    }
    return $null
}

function PanelText($root, [int]$minLen = 400) {
    $best = ""
    foreach ($e in (Descendants $root)) {
        $ct = PropOf $e ([System.Windows.Automation.AutomationElement]::ControlTypeProperty)
        if ($ct -ne [string][System.Windows.Automation.ControlType]::Edit) { continue }
        $vp = $null
        try { $vp = $e.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern) } catch { continue }
        if ($vp -eq $null) { continue }
        $t = $vp.Current.Value
        if ($t -ne $null -and $t.Length -gt $minLen -and $t.Length -gt $best.Length) { $best = $t }
    }
    return $best
}

# The panel holds the report of ONE agent. The TB-expert PPO agent's report is the block
# that mentions "PPO" and NOT "MLP": the MLP-expert variant is a different agent whose
# report also contains the string "PPO", so a plain -match "PPO" is not an identity test.
function PpoTbSection([string]$text) {
    if ($text -eq "") { return "" }
    $lines = $text -split "`r?`n"
    $startIdx = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "PPO\+MCTS" -and $lines[$i] -notmatch "MLP") { $startIdx = $i; break }
    }
    if ($startIdx -lt 0) { return "" }
    $out = @()
    for ($i = $startIdx; $i -lt $lines.Count; $i++) {
        if ($i -gt $startIdx -and $lines[$i] -match "PPO\+MCTS") { break }
        $out += $lines[$i]
    }
    return ($out -join "`n")
}

Write-Host "=== verify_moe_gui_ppo_full ==="
Write-Host ("exe: " + $Exe)

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
try {
    # ---- 1. window ----
    $root = $null
    $dl = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $dl) {
        Start-Sleep -Milliseconds 500
        if ($proc.HasExited) { break }
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
        $root = [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
            [System.Windows.Automation.TreeScope]::Children, $cond)
        if ($root -ne $null) { break }
    }
    Check ($root -ne $null) "main window found"
    if ($root -eq $null) { throw "no window" }

    # ---- 2. select the TB-expert PPO agent ----
    $combo = $null
    $dl2 = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $dl2) {
        Start-Sleep -Milliseconds 1000
        if ($proc.HasExited) { break }
        $combo = FindByObjectName $root "agentComboBox"
        if ($combo -ne $null) { break }
    }
    Check ($combo -ne $null) "agentComboBox found"
    if ($combo -eq $null) { throw "no combo" }

    $pidCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
    $liCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)

    # Two rounds: first the MLP-expert row (cheap net), then the TB row. The widget follows
    # the 对战AI combo, and selecting a cheap agent first means the widget is fed at least
    # once quickly -- so "state=na forever" cannot be blamed on the heavy net alone.
    $rowsPicked = @()
    foreach ($want in @("MLP", "TB")) {
        $target = $null
        $ec = $combo.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern)
        $ec.Expand()
        Start-Sleep -Milliseconds 800
        $and = New-Object System.Windows.Automation.AndCondition($pidCond, $liCond)
        $items = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Descendants, $and)
        foreach ($it in $items) {
            $n = PropOf $it ([System.Windows.Automation.AutomationElement]::NameProperty)
            if ($n -notlike "*PPO*") { continue }
            if ($want -eq "MLP" -and $n -like "*MLP*") { $target = $it; break }
            if ($want -eq "TB" -and $n -notlike "*MLP*") { $target = $it; break }
        }
        if ($target -ne $null) {
            $nm = PropOf $target ([System.Windows.Automation.AutomationElement]::NameProperty)
            try {
                $target.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
                $rowsPicked += $nm
            } catch { Write-Host ("  info  Select() failed: " + $_.Exception.Message) }
        }
        $ec.Collapse()
        Start-Sleep -Milliseconds 1500
    }
    Check ($rowsPicked.Count -gt 0) "selected PPO row(s) in the agent combo" ($rowsPicked -join " | ")

    # ---- 3. press "self check all models" so the report is rebuilt on demand ----
    $allBtn = FindByObjectName $root "selfCheckAllBtn"
    if ($allBtn -ne $null) {
        try { $allBtn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke() } catch { }
        Write-Host "  info  pressed selfCheckAllBtn"
    }

    # ---- 4. poll until the panel describes the TB PPO agent ----
    $text = ""
    $seen = @()
    $dl3 = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $dl3) {
        Start-Sleep -Milliseconds 2000
        if ($proc.HasExited) { break }
        $t = PanelText $root
        if ($t -eq "") { continue }
        $head = ($t -split "`r?`n" | Where-Object { $_ -match "MoE:" } | Select-Object -First 1)
        if ($head -ne $null -and ($seen -notcontains $head)) {
            $seen += $head
            Write-Host ("  info  panel MoE line: " + $head)
        }
        $sec = PpoTbSection $t
        if ($sec -ne "") { $text = $sec; break }
    }
    Check ($text -ne "") "panel text for the TB-expert PPO agent found"

    if ($text -ne "") {
        foreach ($l in ($text -split "`r?`n")) {
            if ($l -match "^MoE|MaxVio|train|Loss-Free|均衡|门控|有效专家") { Write-Host ("    " + $l) }
        }
    }

    # ---- 5. the load widget ----
    # NOTE (2026-10): the widget is fed by the NON-"all" self-check pass and needs the
    # agent INSTANCE to exist; while the TB backbone is in the middle of its first (very
    # slow: ~8 s/move at the current budget) search, the widget still reads `state=na`.
    # This script therefore reports the widget's value as INFORMATION and does not fail on
    # it -- the widget's own assertions live in tools/verify_moe_load_view.ps1 -PpoMoe,
    # which drives a match for that purpose. What this script exists for is the PANEL text
    # (the only place where "does this agent have the balancing fix" is visible at all).
    $w = MoeWidget $root
    Check ($w -ne $null) "MoeLoadView found"
    if ($w -ne $null) {
        $nm = PropOf $w ([System.Windows.Automation.AutomationElement]::NameProperty)
        Write-Host ("  info  widget accessibleName = '" + $nm + "'")
        Write-Host "  info  (state=na here is expected until the agent finishes one move)"
    }
} finally {
    if (-not $proc.HasExited) {
        try { $proc.CloseMainWindow() | Out-Null } catch { }
        Start-Sleep -Seconds 3
        if (-not $proc.HasExited) { try { $proc.Kill() } catch { } }
    }
}

Write-Host ""
if ($script:fail -eq 0) { Write-Host "RESULT: PASS" -ForegroundColor Green; exit 0 }
Write-Host ("RESULT: FAIL (" + $script:fail + ")") -ForegroundColor Red
exit 1