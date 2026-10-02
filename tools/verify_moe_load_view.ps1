# tools/verify_moe_load_view.ps1
#
# Verifies the sparse-MoE expert-load widget (src/moeloadview.{h,cpp}) is really in the
# running chess.exe: the control exists in the accessibility tree, is laid out inside
# the right-hand panel, and reports a well-formed machine-readable load summary.
#
# Why this check exists (2026-10):
#   The MoE load balance is invisible in every other reading. Routing collapse leaves
#   forward/loss/weights completely normal -- the measured extremes were train-side
#   MaxVio 0.963 (baseline, weakest expert 1.1% of traffic) versus 0.009 (with the
#   auxiliary-loss-free bias). The ONLY way to see the difference is this widget, so
#   "does the widget actually show up and get fed" deserves an automated check rather
#   than a human glancing at a screenshot.
#
# How: UIA only -- no focus stealing, no pixel recognition. The control is located by
# its Qt class name ("MoeLoadView"), which is ASCII by construction; all Chinese text
# is read OUT of the accessibility tree at run time, never written into this script.
#
# ASCII only, no BOM: Windows PowerShell decodes a BOM-less .ps1 as ANSI, so a Chinese
# comment or literal here can corrupt the parse of the very code below it. That is not
# hypothetical -- the first version of verify_agent_combo.ps1 did exactly that.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools/verify_moe_load_view.ps1
#   powershell -ExecutionPolicy Bypass -File tools/verify_moe_load_view.ps1 -SelectMoe
#   powershell -ExecutionPolicy Bypass -File tools/verify_moe_load_view.ps1 -SelectMoe -Live
#   powershell -ExecutionPolicy Bypass -File tools/verify_moe_load_view.ps1 -PpoMoe
#
# -PpoMoe (2026-10) selects a **PPO+MCTS** row instead of a SAC one and additionally
# asserts the train/infer split is present ("train=" in the summary). That split is the
# only way to tell "the balancing mechanism is working" from "the widget is drawing
# numbers": the lifetime total mixes in the search forwards, which outnumber the
# training-batch forwards by orders of magnitude. RL::PPO::finalizeMoeBatch() feeds it;
# if that call is ever dropped the widget still looks perfectly healthy while every
# load conclusion read off it becomes wrong. See docs/moe_gate_experiment_2026_10.md
# section 12.
#
# -Live (2026-10) additionally drives a real match and checks the BREATHING HIGHLIGHT data
# path end to end: with the "呼吸高亮" switch OFF (the default) the widget must report
# hl=off and no live fields at all; after ticking the switch it must report live=on with a
# forward counter (live_serial) that ADVANCES while the AI is searching. That is the whole
# point of the lock-free probe -- the whole decision holds m_agentMutex, so a lock-taking
# readout would sit frozen exactly when the highlight is supposed to move.
# See rl/sparse_moe.hpp and docs/moe_gate_experiment_2026_10.md section 11.

param(
    [string]$Exe = "",
    [int]$TimeoutSec = 60,
    # Also select a sparse-MoE agent in the drop-down and assert the widget reports a
    # real snapshot (state=ok / state=totalonly) instead of "not applicable".
    [switch]$SelectMoe,
    # [2026-10] -PpoMoe: same check but for the **PPO+MCTS** rows. Why it needs its own
    # switch rather than riding on -SelectMoe: the PPO family is a different pair of
    # agents (AGENT_PPOMCTS / AGENT_PPOMCTS_MLP) whose load readout was wired later, and
    # the whole point of this switch is that a regression there (e.g. losing the
    # train/infer split that RL::PPO::finalizeMoeBatch provides) shows up as a plain
    # `state=na`, which is indistinguishable from "you selected the wrong agent" unless
    # somebody actually selects it. Implies -SelectMoe.
    [switch]$PpoMoe,
    # Select a sparse-MoE agent, start a match, and require the live route readout to
    # appear and keep advancing (see the header comment).
    [switch]$Live,
    [int]$LiveSeconds = 150
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

function Descendants($root) {
    $out = New-Object System.Collections.ArrayList
    $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
    $stack = New-Object System.Collections.Stack
    $stack.Push($root)
    while ($stack.Count -gt 0) {
        $cur = $stack.Pop()
        [void]$out.Add($cur)
        $child = $walker.GetFirstChild($cur)
        while ($child -ne $null) {
            $stack.Push($child)
            $child = $walker.GetNextSibling($child)
        }
    }
    return $out
}

function PropOf($el, $prop) {
    try {
        $v = $el.GetCurrentPropertyValue($prop)
        if ($v -eq [System.Windows.Automation.AutomationElement]::NotSupported) { return "" }
        return [string]$v
    } catch { return "" }
}

# Qt's Windows UIA bridge fills AutomationId with the QObject objectName path
# ("QApplication.MainWindow.centralwidget.controlWidget.selfPlayBtn"), which is ASCII
# by construction. Matching on the tail is therefore language-independent -- unlike the
# visible button text, which is Chinese and must never be written into this script.
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

# Expand one combo, look for a row naming a sparse-MoE agent, select it.
# Two ASCII constraints pin the row without any Chinese literal:
#   "MoE" -> matching bare "MoE" is NOT enough: the first row that matched it was
#            'PPO+MCTS (AlphaZero, ...MoE+MLP...)', which uses RL::PPO's own MoE and is
#            deliberately not covered by this widget.
#   "SAC" -> restricts it to the SAC+AZ agents (the two whose load readout is wired).
# Returns the selected row's name, or "" when this combo offers no such row / the row did
# not answer Select() (Qt popup rows sometimes do not; the caller verifies and retries).
function SelectMoeRowInCombo($combo, $pidCond, $liCond, $wantPpo) {
    $ec = $null
    try { $ec = $combo.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern) }
    catch { return "" }
    $ec.Expand()
    Start-Sleep -Milliseconds 700
    $and = New-Object System.Windows.Automation.AndCondition($pidCond, $liCond)
    $items = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $and)
    $target = $null
    foreach ($it in $items) {
        $n = PropOf $it ([System.Windows.Automation.AutomationElement]::NameProperty)
        # -PpoMoe picks the PPO rows instead. The two families are told apart by their
        # prefixes ("PPO" vs "SAC"); both carry "MoE" in the row name. When $wantPpo is
        # false the PPO rows MUST be excluded explicitly -- without that, the first
        # "MoE" row on the desk belongs to PPO and the SAC-only assertions below would
        # be applied to the wrong agent (that exact mix-up is why the original selector
        # pinned "SAC" in the first place).
        $isPpo = ($n -match "PPO")
        $isSac = ($n -match "SAC")
        if (($n -match "MoE") -and ($isPpo -eq [bool]$wantPpo) -and ($isSac -ne [bool]$wantPpo)) {
            $target = $it; break
        }
    }
    $got = ""
    if ($target -ne $null) {
        $got = PropOf $target ([System.Windows.Automation.AutomationElement]::NameProperty)
        try {
            $target.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
        } catch {
            Write-Host ("  info  Select() failed on '" + $got + "': " + $_.Exception.Message)
            $got = ""
        }
    }
    $ec.Collapse()
    Start-Sleep -Milliseconds 400
    return $got
}

Write-Host "=== verify_moe_load_view ==="
Write-Host ("exe: " + $Exe)

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
try {
    # ---- 1. wait for the main window ----
    $root = $null
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
        if ($proc.HasExited) { break }
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
        $root = [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
            [System.Windows.Automation.TreeScope]::Children, $cond)
        if ($root -ne $null) { break }
    }
    if ($proc.HasExited) {
        Check $false "chess.exe is running" ("exited early, code " + $proc.ExitCode)
        exit 1
    }
    Check ($root -ne $null) "main window found via UIA"
    if ($root -eq $null) { exit 1 }

    # ---- 2. the load widget must exist in the accessibility tree ----
    # Qt reports QObject::className() as the UIA class name, so "MoeLoadView" is ASCII
    # and stable; matching on it avoids depending on any translated string.
    $all = Descendants $root
    $widget = $null
    foreach ($e in $all) {
        if ((PropOf $e ([System.Windows.Automation.AutomationElement]::ClassNameProperty)) -eq "MoeLoadView") {
            $widget = $e
            break
        }
    }
    Check ($widget -ne $null) "MoeLoadView present in the accessibility tree" `
        ("scanned " + $all.Count + " elements")
    if ($widget -eq $null) { exit 1 }

    # ---- 3. it must be laid out with a usable rectangle (not collapsed / not offscreen) ----
    $r = $widget.Current.BoundingRectangle
    Check (($r.Width -gt 120) -and ($r.Height -gt 60)) "widget has a usable rectangle" `
        ("{0}x{1} at {2},{3}" -f [int]$r.Width, [int]$r.Height, [int]$r.X, [int]$r.Y)
    Check (-not $widget.Current.IsOffscreen) "widget is on screen"

    # ---- 4. the accessibility summary must be well-formed ----
    # The summary lives in accessibleName (QWidget::accessibleName -> UIA Name).
    # Measured, not assumed: with Qt 6.9's Windows UIA bridge the
    # accessibleDescription (UIA HelpText) reads back EMPTY -- the first version of this
    # script checked HelpText and failed with Name set and HelpText blank. Every other
    # verify script in tools/ reads Name for the same reason, so Name is the channel.
    $name = PropOf $widget ([System.Windows.Automation.AutomationElement]::NameProperty)
    $desc = PropOf $widget ([System.Windows.Automation.AutomationElement]::HelpTextProperty)
    Write-Host ("  info  Name      = '" + $name + "'")
    Write-Host ("  info  HelpText  = '" + $desc + "'  (not the channel; informational only)")
    Check ($name.Length -gt 0) "accessibleName is set"

    $mt = [regex]::Match($name, "state=(\S+)")
    Check $mt.Success "accessibleName carries the machine-readable summary" $name
    $st = "state=" + $mt.Groups[1].Value
    Write-Host ("  info  state token = " + $st)
    Check (($st -eq "state=nodata") -or ($st -eq "state=na") -or
           ($st -eq "state=ok") -or ($st -eq "state=totalonly")) `
        "state token is one of the documented values" $st

    # ---- 5. it must sit ABOVE the self-check text box (that is where it was inserted) ----
    $above = $false
    foreach ($e in $all) {
        $ct = PropOf $e ([System.Windows.Automation.AutomationElement]::ControlTypeProperty)
        if ($ct -eq [string][System.Windows.Automation.ControlType]::Edit) {
            $er = $e.Current.BoundingRectangle
            if (($er.Width -gt 100) -and ($er.Y -gt $r.Y)) { $above = $true }
        }
    }
    Check $above "widget is laid out above a multi-line text panel (the self-check view)"

    # ---- 6. optional: switch to a sparse-MoE agent and expect a real snapshot ----
    if ($SelectMoe -or $PpoMoe) {
        Write-Host "  ---- selecting a sparse-MoE agent in the drop-down ----"
        # Combo rows live in a SEPARATE top-level popup window, so they are searched
        # desktop-wide but scoped by our process id -- a desktop-wide search without the
        # pid filter picks up ListItems from every other window on the machine (that is
        # a bug the existing verify_agent_combo.ps1 already paid for; same idiom here).
        $pidCond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
        $liCond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::ListItem)

        # Three combos can name a sparse-MoE agent: `agentComboBox` (the 对战AI selector --
        # the one that actually sets the board's current agent type, and therefore the one
        # the load widget follows) plus the arena's A/B pickers. Selecting in all three is
        # harmless and makes the match itself use MoE agents.
        #
        # Why this used to flake: Qt's popup rows sometimes do not answer
        # SelectionItemPattern.Select() (verify_thinking_ui.ps1 already documented that and
        # clicks with the mouse instead). A round that silently selects nothing leaves the
        # widget at `state=na`, i.e. a product-looking failure caused by the harness. So the
        # selection is now (a) targeted by object name and (b) **verified** against the
        # widget afterwards, with one retry.
        $selRows = @()
        foreach ($objName in @("agentComboBox", "matchAComboBox", "matchBComboBox")) {
            $c = FindByObjectName $root $objName
            if ($c -eq $null) {
                Write-Host ("  info  combo '" + $objName + "' not found")
                continue
            }
            $got = SelectMoeRowInCombo $c $pidCond $liCond $PpoMoe
            if ($got -ne "") {
                $selRows += ($objName + " -> " + $got)
            }
        }
        Check ($selRows.Count -gt 0) "selected a sparse-MoE agent row in at least one combo" `
            ($selRows -join " | ")
        foreach ($s in $selRows) { Write-Host ("  info  " + $s) }

        $nm2 = ""
        foreach ($attempt in 1..2) {
            # Selecting a MoE agent builds its net (seconds to tens of seconds) and the
            # load widget is then refreshed by the self-check worker, which may itself
            # wait on the agent mutex -- so poll generously.
            $dl2 = (Get-Date).AddSeconds(120)
            while ((Get-Date) -lt $dl2) {
                Start-Sleep -Milliseconds 1000
                $w2 = MoeWidget $root
                if ($w2 -eq $null) { continue }
                $nm2 = PropOf $w2 ([System.Windows.Automation.AutomationElement]::NameProperty)
                if ($nm2 -match "state=(ok|totalonly)") { break }
            }
            if ($nm2 -match "state=(ok|totalonly|noinstance)") { break }
            Write-Host ("  info  still '" + $nm2 + "' -> retrying the 对战AI combo selection") `
                -ForegroundColor Yellow
            $c = FindByObjectName $root "agentComboBox"
            if ($c -ne $null) { [void](SelectMoeRowInCombo $c $pidCond $liCond $PpoMoe) }
        }
        # ---- [2026-10] the PPO rows need the agent INSTANCE to exist ----
        # The load snapshot can only be non-empty once the board has built the agent, and
        # the board builds it lazily (first decision). So for -PpoMoe this check presses
        # "start" and waits for the widget to actually report numbers -- otherwise it would
        # assert on `state=na`, which means "no instance yet", not "the readout is broken".
        # That distinction is exactly what the first run of this check got wrong (it failed
        # while the TB backbone was still starting its first, very slow search).
        if ($PpoMoe -and ($nm2 -notmatch "state=(ok|totalonly)")) {
            $sp = FindByObjectName $root "selfPlayBtn"
            if ($sp -ne $null) {
                Write-Host "  info  pressing selfPlayBtn so the agent thinks once (lazy net)"
                try { $sp.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke() } catch { }
                $dlsp = (Get-Date).AddSeconds(240)
                while ((Get-Date) -lt $dlsp) {
                    Start-Sleep -Milliseconds 2000
                    $w3 = MoeWidget $root
                    if ($w3 -eq $null) { continue }
                    $nm2 = PropOf $w3 ([System.Windows.Automation.AutomationElement]::NameProperty)
                    if ($nm2 -match "state=(ok|totalonly)") { break }
                }
            } else {
                Write-Host "  info  selfPlayBtn not found -- cannot build the instance" -ForegroundColor Yellow
            }
        }
        Write-Host ("  info  Name after select = '" + $nm2 + "'")
        # After selecting a sparse-MoE agent the widget must at least STOP saying
        # "na": it must recognise the type. It may legitimately still say
        # "noinstance" -- the SAC+AZ-MoE net is built lazily (on the first decision),
        # so there are no counters until the agent has thought once. Asserting "ok"
        # here would therefore require driving an actual move, which is a different
        # (much heavier) test; recognising the type is what this check guards.
        Check ($nm2 -match "state=(ok|totalonly|noinstance) experts=[0-9]") `
            "after selecting a MoE agent the widget no longer reports 'na'" $nm2
        if ($nm2 -match "state=noinstance") {
            Write-Host ("  info  instance not built yet (lazy net) -- 'ok' needs one move") -ForegroundColor Yellow
        }
        # ---- [2026-10] the PPO family needs one extra assertion: the train/infer split ----
        # Why it is pinned separately: the split is not cosmetic -- it is the ONLY reading
        # that tells "the balancing mechanism is working" from "the widget is drawing
        # numbers" (the lifetime total mixes in the search forwards, which outnumber the
        # training-batch forwards by orders of magnitude). RL::PPO::finalizeMoeBatch()
        # feeds it; drop that call and the widget still looks perfectly healthy while
        # every conclusion read off it becomes wrong. That class of failure never raises
        # anything, so it has to be asserted here.
        # NOTE: ASCII-only on purpose. This file is BOM-less UTF-8 and Windows PowerShell
        # decodes it as ANSI -- a CJK literal in *code* (as opposed to a comment) gets
        # mangled into bytes that break the parse. That is not hypothetical: the first
        # version of this block had a Chinese assertion message and the script would no
        # longer parse at all ("The Try statement is missing its Catch or Finally block").
        if ($PpoMoe) {
            Check ($nm2 -match "train=") `
                "PPO agent summary carries the train field (split is wired)" $nm2
            if ($nm2 -match "state=ok") {
                Check ($nm2 -match "train=(noforward|[0-9])") `
                    "train field is either noforward or a share number (well formed)" $nm2
            }
        }
    }

    # ---- 7. optional: the live (breathing highlight) readout ----
    if ($Live) {
        Write-Host "  ---- live route readout (breathing highlight) ----"
        $w = MoeWidget $root
        Check ($w -ne $null) "MoeLoadView found for the live check"

        # ---- 7a. the switch must be OFF by default, and off must really mean off ----
        $hcb = FindByObjectName $root "moeHighlightCheck"
        Check ($hcb -ne $null) "the highlight switch exists (moeHighlightCheck)"
        if ($hcb -ne $null) {
            $tgl = $hcb.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
            Check ($tgl.Current.ToggleState -eq
                   [System.Windows.Automation.ToggleState]::Off) `
                "the highlight switch is OFF by default" ([string]$tgl.Current.ToggleState)
            $nmOff = PropOf $w ([System.Windows.Automation.AutomationElement]::NameProperty)
            Write-Host ("  info  Name (switch off) = '" + $nmOff + "'")
            Check ($nmOff -match "hl=off") "the summary reports hl=off while the switch is off" $nmOff
            Check ($nmOff -notmatch "live=") `
                "no live fields at all while it is off (off means off, not 'off but still reading')" $nmOff
        }

        # Wait for startup to finish (the button is enabled only from the startupComplete
        # handler) and start a match: a match makes the resident agents think, and every
        # forward publishes into the lock-free probe the widget reads.
        $sp = FindByObjectName $root "selfPlayBtn"
        Check ($sp -ne $null) "selfPlayBtn found by object name (language-independent)"
        $enabled = $false
        if ($sp -ne $null) {
            $dl = (Get-Date).AddSeconds(180)
            while ((Get-Date) -lt $dl) {
                if ($sp.Current.IsEnabled) { $enabled = $true; break }
                Start-Sleep -Milliseconds 500
                $sp = FindByObjectName $root "selfPlayBtn"
                if ($sp -eq $null) { break }
            }
        }
        Check $enabled "startup finished (selfPlayBtn enabled)"
        if ($enabled -and $hcb -ne $null) {
            # ---- 7b. turn the switch ON FIRST, then start the match ----
            # Order matters and was measured: a finished match makes the app block on the
            # end-of-match weight save (~half a gigabyte), during which the GUI thread does
            # not update anything -- so a sampler that only starts after the match has been
            # running for a while can end up reading the same frozen Name for its whole
            # window (that is exactly how this check flaked once). Ticking the switch first
            # and sampling immediately after the match starts puts the window where the AI
            # is certainly thinking.
            $hcb.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Toggle()
            Start-Sleep -Milliseconds 400
            $nmOn = PropOf (MoeWidget $root) ([System.Windows.Automation.AutomationElement]::NameProperty)
            Write-Host ("  info  Name (switch on)  = '" + $nmOn + "'")
            Check ($nmOn -match "hl=on") "after ticking the switch the summary reports hl=on" $nmOn

            $inv = $sp.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
            $inv.Invoke()

            $serials = New-Object System.Collections.ArrayList
            $sawOn = $false
            $sawTop = $false
            $lastName = ""
            $badName = ""
            $dl2 = (Get-Date).AddSeconds($LiveSeconds)
            while ((Get-Date) -lt $dl2) {
                Start-Sleep -Milliseconds 250
                $w2 = MoeWidget $root
                if ($w2 -eq $null) { continue }
                $nm = PropOf $w2 ([System.Windows.Automation.AutomationElement]::NameProperty)
                if ($nm -eq $lastName) { continue }
                $lastName = $nm
                if ($nm -match "live=on") { $sawOn = $true } else { continue }
                if ($nm -match "live_top=") { $sawTop = $true }
                if (($nm -notmatch "state=ok") -and ($badName -eq "")) { $badName = $nm }
                $m = [regex]::Match($nm, "live_serial=(\d+)")
                if ($m.Success) {
                    $v = [int]$m.Groups[1].Value
                    if (($serials.Count -eq 0) -or ($serials[$serials.Count - 1] -ne $v)) {
                        [void]$serials.Add($v)
                    }
                }
                if (($serials.Count -ge 3) -and $sawTop) { break }
            }
            Write-Host ("  info  Name (last)  = '" + $lastName + "'")
            $uniq = ($serials | Sort-Object -Unique).Count
            Write-Host ("  info  distinct live_serial values seen: " + $uniq + "  -> " +
                        (($serials | Select-Object -First 8) -join ","))
            Check $sawOn "live=on was reported (the probe is registered for the current agent)"
            Check $sawTop "live_top= was reported (the widget knows WHICH expert is working)"
            Check ($uniq -ge 3) "live_serial kept ADVANCING (the readout is really live, not frozen)" $uniq
            Check ($badName -eq "") "whenever live=on the snapshot was state=ok (not a stale mix)" $badName

            # ---- 7c. turning it back off must really stop it ----
            $hcb.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Toggle()
            Start-Sleep -Milliseconds 800
            $nmOff2 = PropOf (MoeWidget $root) ([System.Windows.Automation.AutomationElement]::NameProperty)
            Write-Host ("  info  Name (off again) = '" + $nmOff2 + "'")
            Check ($nmOff2 -match "hl=off") "the switch toggles back off" $nmOff2
            Check ($nmOff2 -notmatch "live=") "and the live readout disappears with it" $nmOff2
            # The load snapshot comes from the locked self-check worker; the live readout
            # comes from the lock-free probe. If someone "simplifies" the live path into
            # the locked one, the serial stops advancing while the AI thinks and the
            # check above fails -- that is the regression this switch exists for.
        }
    }

    Write-Host ""
    if ($script:fail -eq 0) {
        Write-Host "RESULT: PASS" -ForegroundColor Green
    } else {
        Write-Host ("RESULT: FAIL (" + $script:fail + " checks)") -ForegroundColor Red
    }
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
exit $script:fail
