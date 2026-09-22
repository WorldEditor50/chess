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
#   powershell ... -File tools/verify_match_ui.ps1 -AIndex 9 -BIndex 0 -Full `
#       -LogFile build\match_ui_app.log     # 顺便核对 "[weights] 保存" 的计时行
#   :: 复现"对弈结束的静默保存 x 后台训练"那个崩溃场景 (2026-09 用户报的):
#   powershell ... -File tools/verify_match_ui.ps1 -TrainIndex 5 -AIndex 5 -BIndex 0 `
#       -Games 1 -Full -LogFile build\match_ui_crash.log
#
# Combo order in the window: 0 = 对战AI, 1 = A方, 2 = B方, 3 = 历史对局.
# Agent order inside each agent combo (= MainWindow::kAgents, 2026-09 起):
#   0 Alpha-Beta, 1 MCTS, 2 PG, 3 DQN, 4 PPO+MCTS (TB 专家),
#   5 PPO+MCTS (MLP 专家), 6 DQN+MCTS, 7 EVAB, 8 SAC+MCTS+AlphaZero,
#   9 SAC+MCTS+AlphaZero (稀疏 MoE + TransformerBlock 专家), 10 DQN+AB.
#   第 5 项是"同一套 PPO+MCTS 实现 + 另一种骨干", 所以 4 与 5 可以直接对弈比较。
#   **插入新 agent 时必须同步改这里的 $shortName 表**(它是按下标取短名的)。
#   注: 9 = SAC+AZ-MoE 每步 16 次模拟只是**搜索**约 175 ms, 一手还要跑一次在线
#   learnBatch, 端到端实测 ~2.3 s/手 (docs/agents_design.md 13.6)。
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
    # "对战AI" 组合框 (0) 选哪个 agent —— 它决定**后台训练的目标**。
    # 设成与 A/B 同一个 agent 就能复现"对弈结束的静默保存 × 后台训练"那个崩溃场景
    # (2026-09 用户报的), 所以它是这个脚本里唯一一个"为了压竞态"而存在的参数。
    [int]$TrainIndex = -1,
    [switch]$KeepOpen,
    # 把 chess.exe 的 qInfo/qWarning 输出重定向到这里 (stdout 会写到 "<LogFile>.out")。
    # 默认不重定向: 保持和以前完全一样的启动方式, 免得改了这个脚本的行为。
    # 需要看 "[weights] 保存 ... : N ms" 这类计时行时才传它。
    [string]$LogFile = "",
    # 打完第一场后**在同一进程里再打一场**, 并断言奖励读数里仍然只有两条线。
    # 这是 C12("每跑一场就多挂两条空线")的端到端钉子: 那个 bug 只在第二场之后才看得见
    # (第一场是 2 条, 第二场变 4 条, 两条永远"暂无")。
    [switch]$TwoMatches
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

# 同上, 但按**名字前缀**找 (文案会改: 例 "环境奖励 (每手累计..." 后面挂了口径说明)。
# 找不到时返回 "" 并**打印一条警告** —— 静默返回空会把"文案改了"伪装成"没数据"。
function Get-DescPrefix([string]$prefix) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::IsControlElementProperty, $true)
    $all = $script:root.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $cond)
    foreach ($t in $all) {
        if ([string]$t.Current.Name -like "$prefix*") {
            return [string]$t.Current.HelpText
        }
    }
    Write-Output ("[warn] 找不到名字以 '{0}' 开头的控件 (文案改了? 这条读数会是空的)" -f $prefix)
    return ""
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

    # 每个 agent 在下拉框里的序号 -> 结果标签里会出现的短名 (见 ChessBoard::agentDisplayName)。
    # 顺序 = MainWindow::kAgents 的顺序 (2026-09 起 PPO+MCTS 的 MLP 专家骨干插在第 5 位):
    #   0 Alpha-Beta, 1 MCTS, 2 Policy Gradient, 3 DQN,
    #   4 PPO+MCTS (TB 专家), 5 PPO+MCTS (MLP 专家), 6 DQN+MCTS, 7 EVAB,
    #   8 SAC+AZ, 9 SAC+AZ-MoE, 10 DQN+AB
    $shortName = @("Alpha-Beta", "MCTS", "Policy Gradient", "DQN",
                   "PPO+MCTS", "PPO+MCTS-MLP", "DQN+MCTS", "EVAB",
                   "SAC+AZ", "SAC+AZ-MoE", "DQN+AB")
    $wantA = ""
    $wantB = ""
    if ($TrainIndex -ge 0) {
        # 组合框 0 = "对战AI": 它决定的不是对弈参赛者, 而是**后台训练的目标**
        # (backgroundTrainLoop 训的是 m_agentType)。把它设成与 A/B 相同的 agent,
        # 就是"对弈结束的静默保存"与"后台训练那一轮"抢同一张网的现场 ——
        # 2026-09 用户报的崩溃正是这个组合 (修复见 ChessBoard::saveCurrentAgentModel)。
        Write-Output ("training target (对战AI) = {0}" -f (Select-ComboItem 0 $TrainIndex))
    }
    if ($AIndex -ge 0) {
        $wantA = $shortName[$AIndex]
        Write-Output ("A side agent = {0}" -f (Select-ComboItem 1 $AIndex))
    }
    if ($BIndex -ge 0) {
        $wantB = $shortName[$BIndex]
        Write-Output ("B side agent = {0}" -f (Select-ComboItem 2 $BIndex))
    }
    # 这一场有没有"可训练 agent" (0=Alpha-Beta, 1=MCTS 没有可训练参数, 其余都有)。
    # 它决定三件事适不适用: 损失曲线读数、静默保存那一行、保存计时日志 —— 都是
    # "没有可训练参数就本来不该出现"的东西, 拿它们当失败会把配置问题报成产品 bug。
    $expectSave = ($AIndex -gt 1) -or ($BIndex -gt 1)

    Set-Games $Games
    Start-Sleep -Milliseconds 300
    Write-Output "games set to $Games"

    $btn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Milliseconds 1200

    $running = ($null -ne (Find-ByName "停止对弈"))
    Write-Output ("match_running = {0}  (button switched to the stop label)" -f $running)

    $final = ""
    # ---- 对局进行中: 奖励曲线的点数必须在涨 ----
    # 用户报过"对弈时奖励曲线没有更新": 原来一局只采一个点, 一局几百手、十几分钟
    # 里曲线一动不动。现在每手一个点, 所以只要对局在走, 读数里的"N 点"就会变大。
    # 这里在对局进行中隔一会儿取一次样, 最后要求"最后一次 > 第一次"。
    $rewardPts = @()
    if ($Full) {
        $deadline = (Get-Date).AddSeconds($TimeoutSec)
        while ((Get-Date) -lt $deadline) {
            foreach ($t in All-Texts) {
                if ($t -like "*共 *局*手*") { $final = $t }
            }
            if ($final -ne "" -and $final -notlike "*进行中*") { break }
            foreach ($t in All-Texts) {
                if ($t -like "奖励*" -and $t -match "(\d+)\s*点") {
                    $rewardPts += [int]$Matches[1]
                    break
                }
            }
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
    # 奖励曲线的标题在 2026-09 换口径时改过 (现在写明了"学习口径")。按**前缀**取,
    # 免得每次改文案就把这条读数变成空字符串 (那时脚本只会"看起来没数据")。
    $rewardDesc = Get-DescPrefix "环境奖励 (每手累计"
    # 曲线内容是画出来的, UIA 读不到数字; 图下面的数字读数标签才是可读的凭据
    $lossText = ""
    $rewardText = ""
    foreach ($t in All-Texts) {
        if ($t -like "损失 *") { $lossText = $t }
        if ($t -like "奖励*") { $rewardText = $t }
    }
    Write-Output ("score label = " + $score)
    # ---- 对局过程中的增长 (取样在对局进行中完成) ----
    if ($rewardPts.Count -ge 2) {
        $grew = ($rewardPts[$rewardPts.Count - 1] -gt $rewardPts[0])
        Write-Output ("reward curve grew during the match = {0}  (n: {1} -> {2}, {3} 次采样)" -f `
            $grew, $rewardPts[0], $rewardPts[$rewardPts.Count - 1], $rewardPts.Count)
        $rewardOk = $rewardOk -and $grew
    } else {
        Write-Output ("reward curve grew during the match = 跳过 (对局太短, 只采到 {0} 次)" -f `
            $rewardPts.Count)
    }
    Write-Output ("game list items = {0} (其中逐局行 {1})" -f $items.Count, $gameLines.Count)
    foreach ($it in $items) { Write-Output ("   | " + $it) }
    Write-Output ("loss readout   = " + $lossText)
    Write-Output ("reward readout = " + $rewardText)
    if ($lossDesc -ne "") { Write-Output ("loss chart   = " + $lossDesc) }
    if ($rewardDesc -ne "") { Write-Output ("reward chart = " + $rewardDesc) }

    # 每局一行明细 (且行里要有奖励信息)
    $listOk = ($gameLines.Count -ge 1) -and ($gameLines[0] -like "*奖励*")
    $scoreOk = ($score -match "\d+\s*:\s*\d+")
    # 奖励曲线每手一个点 (局末再多一个含 ±1 的点) -> 点数至少要赶上总手数。
    # 这是用户那个"对弈时奖励曲线没有更新"的**防回归断言**: 点数≈1 就是 bug 复现。
    $rewardOk = ($rewardText -match "(\d+)\s*点") -and ([int]$Matches[1] -ge 1)
    $rewardPtsFinal = -1
    if ($rewardText -match "(\d+)\s*点") { $rewardPtsFinal = [int]$Matches[1] }
    $plies = -1
    if ($final -match "(\d+)\s*手") { $plies = [int]$Matches[1] }
    $gamesDone = -1
    if ($final -match "共\s*(\d+)\s*局") { $gamesDone = [int]$Matches[1] }
    if ($rewardPtsFinal -ge 0 -and $plies -ge 0 -and $gamesDone -ge 0) {
        # 一手一个点, 每局再多一个"局末含 ±1"的点; 被将死那一手不落子所以每局可能少一个,
        # 于是下界取 (手数 - 局数)。
        $wantPts = $plies - $gamesDone
        $perMoveOk = ($rewardPtsFinal -ge $wantPts)
        Write-Output ("reward points vs plies = {0} 点 / {1} 手 (下界 {2}) -> {3}" -f `
            $rewardPtsFinal, $plies, $wantPts, $perMoveOk)
        $rewardOk = $rewardOk -and $perMoveOk
    }
    # ---- C12 的界面钉子: 奖励读数里只该有**两条**线, 而且不该有"暂无" ----
    # 每开一场就 clearData()+addSeries()x2 的写法会让第 N 场出现 2N 条线 (2N-2 条空的),
    # 读数里就是 "A: 暂无 | B: 暂无 | A: 最新 ... | B: ..." —— 用户在自己的 100 局对弈里
    # 就是这么看到的。每条线的读数末尾是"N 点"或"暂无", 数一数就知道有几条线。
    $seriesChunks = [regex]::Matches($rewardText, "点|暂无").Count
    $noEmpty = ($rewardText -notlike "*暂无*")
    Write-Output ("reward series in readout = {0} 条 (期望 2), 有空线 = {1}" -f `
        $seriesChunks, (-not $noEmpty))
    # 只在真打过点的时候要求"无空线" (一次都没采样过的场次本来就是空线, 不算 bug)
    if ($rewardPtsFinal -ge 1) {
        $rewardOk = $rewardOk -and ($seriesChunks -eq 2) -and $noEmpty
    }
    # ---- [④ 2026-09] 奖励曲线必须**标出口径** ----
    # 奖励曲线现在取 agent 自己的学习口径 (材质 x0.1 + 每步代价 + 终局), 纯搜索 agent
    # (Alpha-Beta / MCTS / EVAB) 没有学习口径、仍是引擎口径 (材质 x1) —— 两个口径差 10 倍,
    # 不标出来两条线就不能直接比大小。曲线名后缀是这件事唯一的提示, 所以在这里钉住:
    # 读数里必须出现 [学习口径] / [引擎口径], 而且**两条线各一次**。
    $learnTags = [regex]::Matches($rewardText, "\[学习口径\]").Count
    $engineTags = [regex]::Matches($rewardText, "\[引擎口径\]").Count
    Write-Output ("reward caliper tags = 学习 {0} / 引擎 {1} (期望合计 2)" -f `
        $learnTags, $engineTags)
    $caliperOk = (($learnTags + $engineTags) -eq 2)
    if (-not $caliperOk) { Write-Output "   [warn] 奖励曲线的口径标签不见了 (曲线名/口径改动后漏了标注?)" }
    $rewardOk = $rewardOk -and $caliperOk
    if ($expectSave) {
        # 可训练的 agent 一定有学习口径; 这条把"标签写反"也钉住 (A/B 两方各一个标签)
        Write-Output ("   (A 方是可训练 agent => 至少应有一个 [学习口径] 标签: {0})" -f `
            ($learnTags -ge 1))
        $rewardOk = $rewardOk -and ($learnTags -ge 1)
    }
    # 损失读数: 会上报损失的 agent 才有数字; 两边都是 Alpha-Beta/MCTS 时读数就是 "-"
    # (没有可训练参数 -> 不上报, 曲线里没有点, 这是**正确**行为, 不能算失败)。
    # 注意 $lossText 是**带前缀**的 ("损失 ..."), 要先把前缀去掉再比 —— 否则
    # "-" 永远匹配不上, 这个跳过分支就是死代码 (第一版就是这么写的)。
    $lossBody = $lossText -replace "^损失\s*", ""
    $lossOk = $true
    if ($lossBody -eq "-" -or $lossBody -eq "") {
        if (-not $expectSave) {
            Write-Output "   (跳过损失读数检查: 两边都没有可训练参数, 本来就不该有曲线)"
        } else {
            $lossOk = $false
        }
    } else {
        $lossOk = ($lossText -match "\d+\s*点")
    }
    # ---- 静默保存: 列表里应出现"已静默保存权重"的行, 且不该有文件对话框的残留 ----
    # 保存跑在后台线程, 而且稀疏 MoE 变体的权重是 3x146 MB, 所以这里要**等**
    # 它出现, 不能在下棋结束的一瞬间就去读列表 (第一版就是这么误报失败的)。
    #
    # 但"该不该有这一行"取决于参赛的是谁: Alpha-Beta 与 MCTS 没有可训练参数、
    # 也就没有权重文件, saveWeightsAfterMatch 会跳过它们。所以两边都是 0/1 的时候
    # **本来就不该有**这一行 —— 不当成失败, 打印跳过原因 (否则脚本会把自己的配置错误
    # 报成产品 bug)。$expectSave 在上面按 A/B 序号算好了。
    $silentSave = $false
    $saveDeadline = (Get-Date).AddSeconds(60)
    while ($expectSave -and -not $silentSave -and (Get-Date) -lt $saveDeadline) {
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
    if (-not $expectSave) {
        Write-Output ("   (跳过静默保存检查: A/B 都是无可训练参数的 agent, 本来就不存权重)")
        $silentSave = $true
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
        if (($a -eq "-") -and (-not $expectSave)) {
            # 没有可训练 agent -> 源控件本来就是空的, 放大窗口也只能是空的,
            # 没有"镜像"可比 (报 False 就是把配置问题当成 bug)
            Write-Output "   (跳过放大窗口数据比对: 这一场没有损失曲线)"
            $largeDataOk = $true
        } else {
            $largeDataOk = ($a -eq $b) -and ($a -ne "-")
            Write-Output ("   large window mirrors the source label = {0}" -f $largeDataOk)
        }
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

    # ---- 可选: 同一进程里再打一场, 钉住 C12("每跑一场就多挂两条空线") ----
    # 那个 bug 在第一场里看不出来 (第一场本来就该是 2 条线), 第二场才会变成 4 条
    # (其中两条永远"暂无")。所以只在 -TwoMatches 时做这一步, 免得日常验证变慢。
    if ($TwoMatches) {
        $btn2 = Find-ByName "开始对弈"
        if ($null -eq $btn2) {
            Write-Output "second match: start button not found -> FAIL"
            $ok = $false
        } else {
            # 先数一下逐局明细的行数: 第二场真的跑了的话, 列表里会多出
            # "—— 第 2 场 ——" + 每局一行 + "本场结束" 至少 3 行。
            # 没有这个证据的话, "第二场只有两条线"也可能是因为**第二场根本没跑**
            # (读数还是第一场留下的) —— 那就是假通过。
            $rowsBefore = (All-ListItems).Count
            $btn2.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
            Start-Sleep -Milliseconds 1200
            $deadline2 = (Get-Date).AddSeconds($TimeoutSec)
            while ((Get-Date) -lt $deadline2) {
                if ($null -eq (Find-ByName "停止对弈")) { break }
                Start-Sleep -Milliseconds 700
            }
            Start-Sleep -Seconds 2
            $rowsAfter = (All-ListItems).Count
            $ran2 = ($rowsAfter -gt $rowsBefore)
            Write-Output ("second match ran = {0}  (逐局明细 {1} -> {2} 行)" -f `
                $ran2, $rowsBefore, $rowsAfter)
            $r2 = ""
            foreach ($t in All-Texts) { if ($t -like "奖励*") { $r2 = $t } }
            $chunks2 = [regex]::Matches($r2, "点|暂无").Count
            $noEmpty2 = ($r2 -notlike "*暂无*")
            Write-Output ("second match reward readout = " + $r2)
            Write-Output ("second match series = {0} 条 (期望 2), 有空线 = {1}" -f `
                $chunks2, (-not $noEmpty2))
            $ok = $ok -and $ran2 -and ($chunks2 -eq 2) -and $noEmpty2
        }
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
        if (-not $expectSave) {
            Write-Output ("   (跳过保存计时检查: 这一场没有可保存权重的 agent)")
            $timed = $true
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
