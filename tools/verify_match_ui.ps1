# tools/verify_match_ui.ps1
#
# End-to-end check of the Agent-vs-Agent arena through the real GUI
# (see docs/agents_design.md section 9). Launches chess.exe, optionally selects
# the two competing agents in the combo boxes, sets the number of games, starts
# the match, and reads the result text back.
#
# It drives the app through Windows UI Automation instead of simulated
# keystrokes, for two reasons learned the hard way:
#
#   * SendKeys only works when the app owns the foreground, and
#     WScript.Shell.AppActivate fails silently when the console has it. UIA
#     InvokePattern / SelectionItemPattern work regardless of focus.
#   * UIA also hands back the widget rectangles and their text, so the script can
#     *assert on the result label* instead of trying to OCR pixels.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File tools/verify_match_ui.ps1
#   powershell ... -File tools/verify_match_ui.ps1 -Games 2 -Full
#   powershell ... -File tools/verify_match_ui.ps1 -AIndex 7 -BIndex 0 -Full
#   powershell ... -File tools/verify_match_ui.ps1 -AIndex 8 -BIndex 0 -Full `
#       -LogFile build\match_ui_app.log     # 顺便核对 "[weights] 保存" 的计时行
#
# Combo order in the window: 0 = 对战AI, 1 = A方, 2 = B方, 3 = 历史对局.
# Agent order inside each agent combo: 0 Alpha-Beta, 1 MCTS, 2 PG, 3 DQN,
# 4 PPO+MCTS, 5 DQN+MCTS, 6 EVAB, 7 SAC+MCTS+AlphaZero,
# 8 SAC+MCTS+AlphaZero with the sparse-MoE / TransformerBlock-expert backbone
# (short name "SAC+AZ-MoE"; 16 simulations per move is ~175 ms of *search* only --
# a full move also runs one online learnBatch, and the measured end-to-end cost is
# ~2.3 s/move, see docs/agents_design.md 13.6).
#
# ASCII only in code; the Chinese literals below are matched against the UI, so
# this file must be saved as UTF-8 **with BOM** for Windows PowerShell to
# decode it correctly.
#
# 注意: 有些编辑器/补丁工具保存 UTF-8 时会**丢掉 BOM** (本仓库踩过一次: 改完这个
# 文件再跑, Windows PowerShell 按 ANSI 解码, 中文串被拆成乱码, 报出来的却是一堆
# "Missing expression after ','" 之类**和真正原因毫无关系**的语法错)。
# 改完请确认前三个字节是 EF BB BF:
#   [System.IO.File]::ReadAllBytes("tools\verify_match_ui.ps1")[0..2]

param(
    [string]$Exe = "",
    [int]$Games = 1,
    [switch]$Full,           # wait for the whole match instead of aborting early
    [int]$TimeoutSec = 300,
    [int]$AIndex = -1,       # agent index for side A (-1 = leave the default)
    [int]$BIndex = -1,       # agent index for side B
    [switch]$KeepOpen,
    # 把 chess.exe 的 qInfo/qWarning 输出重定向到这里 (stdout 会写到 "<LogFile>.out")。
    # 默认不重定向: 保持和以前完全一样的启动方式, 免得改了这个脚本的行为。
    # 需要看 "[weights] 保存 ... : N ms" 这类计时行时才传它。
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path
$exeDir = Split-Path $Exe -Parent

# 从普通 shell 启动时 PATH 里通常没有 Qt 的 bin, 进程会以 0xC0000135
# (STATUS_DLL_NOT_FOUND) 直接退出 —— 表现出来只是"窗口没起来", 很容易被误判成
# 应用启动崩了。所以这里先确认 Qt6Widgets.dll 能找到, 找不到就在常见的 Qt 安装
# 目录里搜一个并前置到 PATH。ctest 那边是用 ENVIRONMENT_MODIFICATION 解决的
# (见 CMakeLists.txt 的 test_match)。
$qtFound = $false
foreach ($d in ($env:PATH -split ';')) {
    if ($d -ne "" -and (Test-Path (Join-Path $d "Qt6Widgets.dll"))) {
        $qtFound = $true
        break
    }
}
if (-not $qtFound) {
    $hit = Get-ChildItem -Path "C:\Qt\*\msvc*\bin\Qt6Widgets.dll" -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if ($null -ne $hit) {
        $env:PATH = $hit.DirectoryName + ";" + $env:PATH
        Write-Output ("Qt bin prepended = {0}" -f $hit.DirectoryName)
    } else {
        Write-Output "WARNING: Qt6Widgets.dll not found on PATH and not under C:\Qt\*\msvc*\bin"
    }
}

function Find-ByName([string]$name) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
    return $script:root.FindFirst(
        [System.Windows.Automation.TreeScope]::Descendants, $cond)
}

function All-Texts {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Text)
    $all = $script:root.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $cond)
    $out = @()
    foreach ($t in $all) { $out += $t.Current.Name }
    return $out
}

# 按 Name 找控件并读它的 HelpText (AccessibleDescription)。
# 曲线控件的内容全是画出来的, UIA 读不到数字 —— 它把状态写进
# AccessibleDescription, 这里就当"曲线的仪表读数"用。
function Get-Desc([string]$name) {
    $e = Find-ByName $name
    if ($null -eq $e) { return "" }
    return [string]$e.Current.HelpText
}

# 界面上所有列表项的文字 (逐局明细在 QListWidget 里)
function All-ListItems {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    $all = $script:root.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $cond)
    $out = @()
    foreach ($t in $all) { $out += $t.Current.Name }
    return $out
}

function Set-Games([double]$v) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Spinner)
    $spins = $script:root.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $cond)
    if ($spins.Count -lt 1) { throw "no spinner found" }
    $pat = $null
    if (-not $spins.Item(0).TryGetCurrentPattern(
            [System.Windows.Automation.RangeValuePattern]::Pattern, [ref]$pat)) {
        throw "games spinner has no RangeValue pattern"
    }
    $pat.SetValue($v)
}

# Select item n of combo m.
#
# SelectionItemPattern.Select() on a Qt combo popup item silently does nothing:
# the first version of this script printed the item's name, looked like it had
# worked, and then ran the *default* pairing (Alpha-Beta vs EVAB) while claiming
# to test SAC+AZ. So now the item is clicked with the mouse and the combo's own
# current value is verified before returning.
function Select-ComboItem([int]$comboIndex, [int]$itemIndex) {
    Add-Type -MemberDefinition @'
[DllImport("user32.dll")] public static extern void mouse_event(uint f,uint dx,uint dy,uint d,int e);
'@ -Name MU -Namespace W32 -PassThru | Out-Null

    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ComboBox)
    $combos = $script:root.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $cond)
    if ($combos.Count -le $comboIndex) { throw "combo $comboIndex not found" }
    $cb = $combos.Item($comboIndex)

    $expand = $null
    $canExpand = $cb.TryGetCurrentPattern(
        [System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$expand)
    if (-not $canExpand) { throw "combo $comboIndex cannot be expanded" }
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
    if ($r.Width -gt 0 -and $r.Height -gt 0) {
        $x = [int]($r.X + $r.Width / 2)
        $y = [int]($r.Y + $r.Height / 2)
        [System.Windows.Forms.Cursor]::Position =
            New-Object System.Drawing.Point($x, $y)
        Start-Sleep -Milliseconds 150
        [W32.MU]::mouse_event(0x0002, 0, 0, 0, 0)
        Start-Sleep -Milliseconds 60
        [W32.MU]::mouse_event(0x0004, 0, 0, 0, 0)
    }
    Start-Sleep -Milliseconds 500
    if ($cb.Current.ExpandCollapseState -ne
        [System.Windows.Automation.ExpandCollapseState]::Collapsed) {
        $expand.Collapse()
    }
    # 收起弹窗要等一下再收工: 连续选两个下拉框时, 如果第一个弹窗还没收起,
    # 第二次点击会打在那个仍然打开的弹窗上 (实测把 A 方选成了弹窗里同一位置的
    # 另一个 agent —— 结果标签的校验抓到了它, 这就是那一步校验存在的意义)。
    for ($w = 0; $w -lt 10; $w++) {
        if ($cb.Current.ExpandCollapseState -eq
            [System.Windows.Automation.ExpandCollapseState]::Collapsed) { break }
        Start-Sleep -Milliseconds 200
    }
    Start-Sleep -Milliseconds 400

    # 注意: Qt 的 QComboBox 在 UIA 里 Name 是**空的**, 读不回当前选择。
    # 所以这里只返回"想选的那一项", 真正的校验在下面用**对弈结果标签**做
    # (结果标签里会打印两个 agent 的名字)。
    return $want
}

$proc = $null
if ([string]::IsNullOrEmpty($LogFile)) {
    $proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
} else {
    # Start-Process 不允许 stdout 和 stderr 用同一个文件, 所以用两个。
    # Qt 的日志走 stderr, 关心的计时行都在 "<LogFile>" 里。
    $proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru `
        -RedirectStandardError $LogFile -RedirectStandardOutput ($LogFile + ".out")
}
Write-Output "launched chess.exe pid=$($proc.Id)"
# 等主窗口真的出来 (固定 sleep 在机器忙的时候会不够, 早先就因此报过
# "hwnd cannot be IntPtr.Zero"; 这里改成最多等 30 s)
$hwnd = [IntPtr]::Zero
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    $q = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($null -eq $q) { throw "chess.exe exited early (exit code $($q.ExitCode))" }
    $hwnd = $q.MainWindowHandle
    if ($hwnd -ne [IntPtr]::Zero) { break }
}
Write-Output ("main window ready after ~{0:N1} s" -f ($i * 0.5))

$script:root = $null
try {
    $script:root = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)
    if ($null -eq $script:root) { throw "no main window handle" }

    $btn = Find-ByName "开始对弈"
    if ($null -eq $btn) { throw "start button not found" }
    # 窗口出现 != 可以交互: startupLoad() 完成之前所有控件都是 disabled 的
    # (这时 SetValue 会抛 "operation is not allowed on a nonenabled element")。
    # 所以等到按钮真的 enabled 再往下走。
    $ready = $false
    $k = 0
    for ($k = 0; $k -lt 60; $k++) {
        if ($btn.Current.IsEnabled) { $ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    Write-Output ("start button enabled = {0} (waited ~{1:N1} s)" -f $ready, ($k * 0.5))
    if (-not $ready) { throw "start button never became enabled (startup load failed?)" }
    Write-Output ("start button rect = {0}" -f $btn.Current.BoundingRectangle)

    # 每个 agent 在下拉框里的序号 -> 结果标签里会出现的短名 (见 ChessBoard::agentDisplayName)
    $shortName = @("Alpha-Beta", "MCTS", "Policy Gradient", "DQN",
                   "PPO+MCTS", "DQN+MCTS", "EVAB", "SAC+AZ", "SAC+AZ-MoE")
    $wantA = ""
    $wantB = ""
    if ($AIndex -ge 0) {
        $wantA = $shortName[$AIndex]
        Write-Output ("A side agent = {0}" -f (Select-ComboItem 1 $AIndex))
    }
    if ($BIndex -ge 0) {
        $wantB = $shortName[$BIndex]
        Write-Output ("B side agent = {0}" -f (Select-ComboItem 2 $BIndex))
    }

    Set-Games $Games
    Start-Sleep -Milliseconds 300
    Write-Output "games set to $Games"

    $btn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Milliseconds 1200

    $running = ($null -ne (Find-ByName "停止对弈"))
    Write-Output ("match_running = {0}  (button switched to the stop label)" -f $running)

    $final = ""
    if ($Full) {
        $deadline = (Get-Date).AddSeconds($TimeoutSec)
        while ((Get-Date) -lt $deadline) {
            foreach ($t in All-Texts) {
                if ($t -like "*共 *局*手*") { $final = $t }
            }
            if ($final -ne "" -and $final -notlike "*进行中*") { break }
            Start-Sleep -Milliseconds 700
        }
    } else {
        Start-Sleep -Seconds 3
        $stop = Find-ByName "停止对弈"
        if ($null -ne $stop) {
            $stop.GetCurrentPattern(
                [System.Windows.Automation.InvokePattern]::Pattern).Invoke()
            Write-Output "invoked stop"
        }
        Start-Sleep -Seconds 4
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            foreach ($t in All-Texts) {
                if ($t -like "*共 *局*手*") { $final = $t }
            }
            if ($final -ne "") { break }
            Start-Sleep -Milliseconds 500
        }
    }

    Write-Output ("result label = " + $final)
    foreach ($t in All-Texts) {
        if ($t -like "AI思考时间:*" -or $t -like "探索+预训练:*") {
            Write-Output ("   " + $t)
        }
    }

    # ---- 新增: 实时比分 / 逐局明细 / 两条曲线 ----
    $score = ""
    $scoreEl = Find-ByName "当前比分:*"
    if ($null -ne $scoreEl) { $score = [string]$scoreEl.Current.Name }
    if ($score -eq "") {
        foreach ($t in All-Texts) {
            if ($t -like "当前比分:*" -or $t -like "最终比分:*") { $score = $t }
        }
    }
    $items = All-ListItems
    # 只看"逐局明细"里的行 (第 N 局: ...): 存权重对话框里也会冒出 ListItem
    $gameLines = @()
    foreach ($it in $items) { if ($it -like "第*局*") { $gameLines += $it } }
    $lossDesc = Get-Desc "训练损失 (每完成一次在线训练一个点)"
    $rewardDesc = Get-Desc "每局环境奖励 (吃子 + 终局 ±1, 走子方视角)"
    # 曲线内容是画出来的, UIA 读不到数字; 图下面的数字读数标签才是可读的凭据
    $lossText = ""
    $rewardText = ""
    foreach ($t in All-Texts) {
        if ($t -like "损失 *") { $lossText = $t }
        if ($t -like "奖励 *") { $rewardText = $t }
    }
    Write-Output ("score label = " + $score)
    Write-Output ("game list items = {0} (其中逐局行 {1})" -f $items.Count, $gameLines.Count)
    foreach ($it in $items) { Write-Output ("   | " + $it) }
    Write-Output ("loss readout   = " + $lossText)
    Write-Output ("reward readout = " + $rewardText)
    if ($lossDesc -ne "") { Write-Output ("loss chart   = " + $lossDesc) }
    if ($rewardDesc -ne "") { Write-Output ("reward chart = " + $rewardDesc) }

    # 每局一行明细 (且行里要有奖励信息)
    $listOk = ($gameLines.Count -ge 1) -and ($gameLines[0] -like "*奖励*")
    $scoreOk = ($score -match "\d+\s*:\s*\d+")
    # 奖励曲线每局两个点 (A/B 各一个) -> 读数里必须出现"N 点"
    $rewardOk = ($rewardText -match "(\d+)\s*点") -and ([int]$Matches[1] -ge 1)
    # 损失读数: 会上报损失的 agent 才有数字, 不上报的显示"暂无"
    $lossOk = ($lossText -like "*暂无*") -or ($lossText -match "\d+\s*点")
    # ---- 静默保存: 列表里应出现"已静默保存权重"的行, 且不该有文件对话框的残留 ----
    # 保存跑在后台线程, 而且稀疏 MoE 变体的权重是 3x146 MB (十几秒), 所以这里要**等**
    # 它出现, 不能在下棋结束的一瞬间就去读列表 (第一版就是这么误报失败的)。
    $silentSave = $false
    $saveDeadline = (Get-Date).AddSeconds(60)
    while (-not $silentSave -and (Get-Date) -lt $saveDeadline) {
        foreach ($it in (All-ListItems)) {
            if ($it -like "*静默保存权重*") { $silentSave = $true; Write-Output ("   | " + $it) }
        }
        if (-not $silentSave) { Start-Sleep -Milliseconds 500 }
    }
    if (-not $silentSave) {
        foreach ($it in $items) {
            if ($it -like "*静默保存权重*") { $silentSave = $true }
        }
    }
    # 以前对弈结束会弹一个**模态**文件对话框, 它的文件列表会混进 ListItem 里
    # (目录项、.dat 文件名...)。现在静默保存, 这些"杂物"必须一条都没有。
    $stray = @()
    foreach ($it in $items) {
        if ($it -notlike "第*局*" -and $it -notlike "——*" -and $it -notlike "(*" -and $it -notlike "思考耗时*") { $stray += $it }
    }
    Write-Output ("silent weight save line   = {0}" -f $silentSave)
    Write-Output ("stray list items (dialogs)= {0}" -f $stray.Count)
    foreach ($s in $stray) { Write-Output ("   ? " + $s) }

    Write-Output ("list has per-game lines = {0}" -f $listOk)
    Write-Output ("score shows a ratio     = {0}" -f $scoreOk)
    Write-Output ("reward chart has points = {0}" -f $rewardOk)
    Write-Output ("loss readout present    = {0}" -f $lossOk)

    $back = ($null -ne (Find-ByName "开始对弈"))
    Write-Output ("button back to start label = {0}" -f $back)

    # ---- 双击曲线 -> 放大窗口 ----
    # 放大窗口是**独立的顶层窗口**, 所以要在桌面根节点下找, main window 的子节点里
    # 是找不到它的。这里顺便读一下窗口里的"读数"标签, 确认放大窗口也拿到了数据。
    #
    # 顺序很重要: 对弈结束后 offerSaveWeights() 会弹一个**模态**文件对话框, 它会吃掉
    # 之后所有的鼠标事件 (第一次跑这个检查时双击就打在它上面了)。先按几次 ESC 把挂起
    # 的对话框关掉, 再做双击。
    for ($i = 0; $i -lt 4; $i++) {
        [System.Windows.Forms.SendKeys]::SendWait("{ESC}")
        Start-Sleep -Milliseconds 300
    }
    Start-Sleep -Milliseconds 600

    $largeOk = $false
    $largeText = ""
    $chartEl = Find-ByName "训练损失 (每完成一次在线训练一个点)"
    if ($null -ne $chartEl) {
        $r = $chartEl.Current.BoundingRectangle
        if ($r.Width -gt 4 -and $r.Height -gt 4) {
            [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point(
                ([int]($r.X + $r.Width / 2)), ([int]($r.Y + $r.Height / 2)))
            Start-Sleep -Milliseconds 200
            for ($c = 0; $c -lt 2; $c++) {
                [W32.MU]::mouse_event(0x0002, 0, 0, 0, 0)
                Start-Sleep -Milliseconds 40
                [W32.MU]::mouse_event(0x0004, 0, 0, 0, 0)
                Start-Sleep -Milliseconds 60
            }
            Start-Sleep -Milliseconds 800
            $deskCond = New-Object System.Windows.Automation.PropertyCondition(
                [System.Windows.Automation.AutomationElement]::NameProperty,
                "训练损失 (放大)")
            # 放大窗口的**父窗口**是主窗口 (Qt 的 QDialog(parent)), Windows 把它当
            # "被拥有的顶层窗口", UIA 把它挂在主窗口下面而不是桌面根节点的 Children 里
            # —— 必须用 Descendants 找 (实测: Children=False, Descendants=True)。
            $dlg = [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
                [System.Windows.Automation.TreeScope]::Descendants, $deskCond)
            if ($null -ne $dlg) {
                $largeOk = $true
                $txtCond = New-Object System.Windows.Automation.PropertyCondition(
                    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
                    [System.Windows.Automation.ControlType]::Text)
                foreach ($t in $dlg.FindAll(
                        [System.Windows.Automation.TreeScope]::Descendants, $txtCond)) {
                    if ($t.Current.Name -like "读数*") { $largeText = $t.Current.Name }
                }
                # 用 WindowPattern 关掉它 (不依赖焦点)
                $wp = $null
                if ($dlg.TryGetCurrentPattern(
                        [System.Windows.Automation.WindowPattern]::Pattern, [ref]$wp)) {
                    $wp.Close()
                }
                Start-Sleep -Milliseconds 500
            }
        }
    }
    Write-Output ("double-click opens a large chart window = {0}" -f $largeOk)
    if ($largeText -ne "") { Write-Output ("   large window readout = " + $largeText) }
    # 放大窗口要和源控件**逐字一致** (去掉前缀"读数"/"损失"再比): 这同时验证了
    # "窗口打开了"和"数据同步过来了"两件事, 而且源控件没数据时也不会假失败。
    $largeDataOk = $false
    if ($largeText -ne "") {
        $a = $largeText -replace "^读数\s*", ""
        $b = $lossText -replace "^损失\s*", ""
        $largeDataOk = ($a -eq $b) -and ($a -ne "-")
        Write-Output ("   large window mirrors the source label = {0}" -f $largeDataOk)
    }
    Write-Output ("large window has data = {0}" -f $largeDataOk)

    # dismiss the result box and any weight-save dialog
    for ($i = 0; $i -lt 8; $i++) {
        [System.Windows.Forms.SendKeys]::SendWait("{ESC}")
        Start-Sleep -Milliseconds 350
    }
    Start-Sleep -Seconds 1
    $alive = ($null -ne (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue))
    Write-Output ("app_still_alive = {0}" -f $alive)

    $ok = $running -and $alive -and ($final -ne "") -and $listOk -and $scoreOk -and $rewardOk -and $lossOk -and $largeOk -and $largeDataOk -and $silentSave -and ($stray.Count -eq 0)

    # 选择是否真的生效: 用结果标签里的 agent 名字核对 (这是唯一可靠的证据)。
    # 必须按**完整词**匹配: "SAC+AZ" 是 "SAC+AZ-MoE" 的前缀, 用 -like "*$want*"
    # 会让 "想选 SAC+AZ、实际选了 SAC+AZ-MoE" 这种情况假通过。
    if ($wantA -ne "") {
        $hitA = ($final -match ("(^|\s)" + [regex]::Escape($wantA) + "(\s|$)"))
        Write-Output ("A side in result = {0} (want '{1}')" -f $hitA, $wantA)
        $ok = $ok -and $hitA
    }
    if ($wantB -ne "") {
        $hitB = ($final -match ("(^|\s)" + [regex]::Escape($wantB) + "(\s|$)"))
        Write-Output ("B side in result = {0} (want '{1}')" -f $hitB, $wantB)
        $ok = $ok -and $hitB
    }

    # ---- 可选的日志校验: 传了 -LogFile 就顺手核对静默保存的**计时** ----
    # 静默保存跑在后台线程, 界面上只能看到"存完了"这一行, 看不到花了多久。
    # 稀疏 MoE 那三个权重是 3x146 MB, 十几秒的写入如果没人量, 下次再动
    # 保存路径就不知道是快了还是慢了 —— 所以让它自己报一行。
    if (-not [string]::IsNullOrEmpty($LogFile) -and (Test-Path $LogFile)) {
        $saveLines = @()
        foreach ($ln in (Get-Content $LogFile)) {
            if ($ln -like "*[weights]*") { $saveLines += $ln; Write-Output ("   log | " + $ln) }
        }
        $timed = $false
        # 只认**保存**那一行: 启动时的 "[weights] PG: 167 ms" 也是 [weights] 开头、
        # 也带 ms, 拿它去满足这个检查就会"对局还没打完也报通过"(第一版就是这么错的)。
        foreach ($ln in $saveLines) {
            if ($ln -like "*保存*" -and $ln -match "\d+\s*ms") { $timed = $true }
        }
        Write-Output ("save timing logged = {0}" -f $timed)
        $ok = $ok -and $timed
    }

    Write-Output ""
    if ($ok) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }
}
finally {
    if (-not $KeepOpen) {
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    }
}
