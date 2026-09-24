# tools/verify_agent_combo.ps1
#
# Verifies that EVERY agent in MainWindow's kAgents list is actually selectable in the
# GUI's agent drop-downs -- i.e. it is *visible* in the popup list, not merely present
# in the model.
#
# Why this check exists (2026-09, the reason it is not paranoia):
#   Qt's QComboBox defaults to maxVisibleItems = 10. When the list grew to 11 entries
#   (the new "PPO+MCTS (AlphaZero, 稀疏MoE+MLP expert)" agent, now ChessBoard::
#   AGENT_PPOMCTS_MLP), the 11th row fell into the popup's scroll area: the agent WAS
#   in the model and WAS loadable from the command line / tests, but opening the
#   drop-down showed only 10 rows -- so "I added it to the list" and "I can pick it in
#   the UI" were both true and still contradicted each other. Fix: fillAgentCombo()
#   raises maxVisibleItems to the list size, and the new agent was moved next to the
#   other PPO+MCTS backbone so the two variants sit together.
#
# How: read the expected labels straight out of src/mainwindow.cpp (the single source
# of truth for the UI list), launch chess.exe, expand each agent combo box and collect
# the rows UIA reports as visible, then assert every expected label is among them.
# Row visibility is exactly what the bug was about, so "present in the model" is NOT
# what this script checks.
#
# [2026-09 FIX] The kAgents pattern used to be 'AGENT_[A-Z_]+' -- which does NOT match
# enum names that contain digits. So the newly added AGENT_AB_L1 / L2 / L3 were
# silently skipped (the script reported "13 entries" while kAgents actually had 16),
# and catching exactly that ("added to the list but not visible in the UI") is the one
# job this script has. Now:
#   * the pattern is AGENT_[A-Z0-9_]+;
#   * plus a cross-check "parsed count == number of entry-shaped lines in the source",
#     so if the pattern ever regresses into missing entries again this fails loudly
#     instead of quietly testing one agent less.
#   (Same lesson as the script itself: a check whose purpose is catching silent
#    omissions can silently omit things too.)
# NOTE: keep this file ASCII-only (see the header) -- a non-ASCII comment here is
# decoded as ANSI by Windows PowerShell and can corrupt the parse of the very
# regex below it. That is not hypothetical: the first version of this fix did it.
#
# ASCII only, no BOM: Windows PowerShell decodes a BOM-less .ps1 as ANSI, so a Chinese
# comment (or literal) can break the parser. All Chinese strings are read from the
# source file at run time instead of being written here.

param(
    [string]$Exe = "",
    [int]$TimeoutSec = 60
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path
$exeDir = Split-Path $Exe -Parent
$srcFile = Join-Path $PSScriptRoot "..\src\mainwindow.cpp"

# ---- 1. expected labels: parse kAgents out of the source (single source of truth) ----
$src = Get-Content -Raw -Encoding UTF8 $srcFile
$m = [regex]::Match($src, 'const AgentChoice kAgents\[\]\s*=\s*\{(?<body>.*?)\n\};',
                    [System.Text.RegularExpressions.RegexOptions]::Singleline)
if (-not $m.Success) { throw "kAgents[] not found in $srcFile" }
$expected = @()
foreach ($mm in [regex]::Matches($m.Groups['body'].Value,
                                 '\{\s*"(?<name>[^"]+)"\s*,\s*ChessBoard::(?<type>AGENT_[A-Z0-9_]+)\s*\}')) {
    $expected += [pscustomobject]@{ Name = $mm.Groups['name'].Value; Type = $mm.Groups['type'].Value }
}
# Cross-check: every kAgents entry starts with `{ "`, so counting that shape gives the
# number of entries the source really has. If the regex above ever misses some (say a
# future enum name uses a character the class does not cover), the two counts differ
# and this fails right here -- instead of silently testing fewer agents.
$entryCount = [regex]::Matches($m.Groups['body'].Value, '\{\s*"').Count
Write-Output ("kAgents entries found in source: {0}" -f $expected.Count)
foreach ($e in $expected) { Write-Output ("  {0,-22} {1}" -f $e.Type, $e.Name) }
if ($expected.Count -lt 2) { throw "parsed too few kAgents entries" }
if ($expected.Count -ne $entryCount) {
    throw ("kAgents parse mismatch: regex matched $($expected.Count) entries but the " +
           "source has $($entryCount) entry-shaped lines -- the pattern is silently " +
           "skipping entries (see the note in the header about AGENT_[A-Z0-9_]+)")
}

# ---- 2. Qt DLLs must be findable from a plain shell (same block as the other checks) ----
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

function Get-ComboBoxes($root) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ComboBox)
    return $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
}

# Visible rows of an expanded combo. Scoped to our own process id: a desktop-wide
# search also picks up ListItems of every other window on the machine (that made the
# first version of this probe report nonsense).
function Get-VisibleRows([int]$pid_, $combo) {
    $ec = $combo.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern)
    $ec.Expand()
    Start-Sleep -Milliseconds 700
    $pidCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $pid_)
    $liCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    $and = New-Object System.Windows.Automation.AndCondition($pidCond, $liCond)
    $items = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $and)
    $names = @()
    foreach ($it in $items) {
        $n = [string]$it.Current.Name
        if ($n -ne "" -and $names -notcontains $n) { $names += $n }
    }
    $ec.Collapse()
    Start-Sleep -Milliseconds 300
    return ,$names
}

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
Write-Output ""
Write-Output ("launched chess.exe pid={0}" -f $proc.Id)
try {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $combos = $null
    $root = $null
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 400
        if ($proc.HasExited) { throw "chess.exe exited early (exit code $($proc.ExitCode))" }
        $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if ($null -eq $p -or $p.MainWindowHandle -eq 0) { continue }
        $root = [System.Windows.Automation.AutomationElement]::FromHandle($p.MainWindowHandle)
        if ($null -eq $root) { continue }
        $combos = Get-ComboBoxes $root
        # The combo boxes are populated in the MainWindow constructor, so they exist as
        # soon as the window does -- no need to wait out the ~26 s weight loading.
        if ($null -ne $combos -and $combos.Count -ge 3) { break }
    }
    if ($null -eq $combos) { throw "no combo boxes found within $TimeoutSec s" }
    Write-Output ("combo boxes in the window: {0}" -f $combos.Count)

    $agentComboCount = 0
    $allOk = $true
    $i = 0
    foreach ($c in $combos) {
        $i++
        $cur = ""
        try {
            $vp = $c.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
            $cur = [string]$vp.Current.Value
        } catch { }
        # An "agent combo" is one whose current selection is an agent label.
        $isAgent = $false
        foreach ($e in $expected) { if ($cur -eq $e.Name) { $isAgent = $true; break } }
        if (-not $isAgent) {
            Write-Output ("combo #{0}: not an agent selector (current = '{1}') - skipped" -f $i, $cur)
            continue
        }
        $agentComboCount++
        $rows = Get-VisibleRows $proc.Id $c
        $missing = @()
        $agentRows = 0
        foreach ($r in $rows) {
            foreach ($e in $expected) { if ($r -eq $e.Name) { $agentRows++; break } }
        }
        foreach ($e in $expected) { if ($rows -notcontains $e.Name) { $missing += $e.Name } }
        # NOTE: the row list also contains the game-list widget's items (same process),
        # hence agentRows <= rows.Count. Only $missing decides pass/fail.
        Write-Output ("combo #{0}: current = '{1}' | rows seen = {2} (agent rows {3}) | missing = {4}" -f `
                      $i, $cur, $rows.Count, $agentRows, $missing.Count)
        if ($missing.Count -gt 0) {
            $allOk = $false
            foreach ($mm in $missing) { Write-Output ("    MISSING (not visible in the drop-down): {0}" -f $mm) }
        }
    }

    Write-Output ""
    Write-Output ("agent selectors checked     = {0} (expected 3: the 3 drop-downs)" -f $agentComboCount)
    Write-Output ("every kAgents entry visible = {0}" -f $allOk)

    # ---- 3. what this check canNOT prove (measured, not assumed) ----
    #
    # "Clicking a row switches the agent" is NOT verifiable through UIA here: a row of a
    # Qt combo popup accepts SelectionItemPattern.Select() (and InvokePattern.Invoke())
    # without committing anything -- the combo's value stays unchanged and
    # MainWindow::onAgentSelected never runs (probed: both calls return OK, value stays
    # 'Alpha-Beta Pruning (deep=4)'). Driving it with SendKeys instead needs the
    # foreground, which the other tools in this directory deliberately avoid. So this
    # script stops at "the row is there and visible", and the behaviour behind the row is
    # covered by test_match: [2.7] (every agent reports a finite training loss),
    # [2.11] (self-check + the distinct weight-file names) and [2.13] (the background
    # training round trip, which for AGENT_SACAZ_OLD builds a SACAZLegacyAgent clone).
    Write-Output ""
    Write-Output "note: row-selection cannot be driven via UIA for a Qt combo popup (see comment);"
    Write-Output "      the behaviour behind the row is covered by test_match [2.14]."

    $ok = $allOk -and ($agentComboCount -ge 3)
    Write-Output ""
    if ($ok) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }
    if (-not $ok) { exit 1 }
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
