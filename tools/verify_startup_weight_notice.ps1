# tools/verify_startup_weight_notice.ps1
#
# 目的 (2026-09 用户报障 "点击开局模型未载入"):
#   确认"没有载入模型"这件事**在界面上看得见** —— 它被挂在"对局列表"最上面那一行。
#
# 为什么单独一个脚本而不是并进 verify_match_ui.ps1: 后者会跑完整场对弈 (分钟级),
#   而这里只需要"启动 -> 读列表"这一步 (秒级), 而且要在**干净目录**里跑 (weight 可能为空)。
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File tools/verify_startup_weight_notice.ps1
#   powershell ... -File tools/verify_startup_weight_notice.ps1 -Exe <path to chess.exe>
#
# 编码约定 (与 verify_agent_combo.ps1 **相反**, 别照抄那一个):
#   本文件里有中文**字面量** (要跟界面上的文字比对), 所以必须存成
#   **UTF-8 with BOM**。无 BOM 时 Windows PowerShell 按 ANSI 解码, 中文被拆成乱码,
#   报出来的却是 "The string is missing the terminator" 这种与真正原因无关的语法错
#   (本文件第一版就是这么坏的: 205 个非 ASCII 字符 + 无 BOM)。
#   核对: [System.IO.File]::ReadAllBytes("<file>")[0..2] 应为 EF BB BF。

param(
    [string]$Exe = "",
    [int]$TimeoutSec = 90
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path

# ---- 在**干净目录**里跑: 那里没有 weights/ (权重是运行期产物, 见 .gitignore) ----
$sandbox = Join-Path $env:TEMP ("chess_startup_probe_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $sandbox | Out-Null
Copy-Item $Exe (Join-Path $sandbox "chess.exe") -Force
Write-Output ("sandbox = {0}" -f $sandbox)

$root = [System.Windows.Automation.AutomationElement]::RootElement
function Find-ByName([string]$name) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
    $w = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
    if ($null -eq $w) {
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::AutomationIdProperty, $name)
        $w = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
    }
    return $w
}
function All-ListItems {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    return $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
}

$proc = Start-Process -FilePath (Join-Path $sandbox "chess.exe") -WorkingDirectory $sandbox -PassThru
Write-Output ("launched pid={0}" -f $proc.Id)
Start-Sleep -Seconds 12   # 启动加载 (无权重时很快; 有权重时这里可能要 10 s+)

# 等"开始对弈"可用 = startupComplete 已发出 = 扫描已结束 (自检面板/列表才是终值)
$deadline = (Get-Date).AddSeconds($TimeoutSec)
$ready = $false
while ((Get-Date) -lt $deadline) {
    $b = Find-ByName "开始对弈"
    if ($null -ne $b -and $b.Current.IsEnabled) { $ready = $true; break }
    Start-Sleep -Milliseconds 400
}
Write-Output ("startup complete (start button enabled) = {0}" -f $ready)

$items = All-ListItems
Write-Output ("list items on the desktop = {0}" -f $items.Count)
$weightLine = ""
for ($i = 0; $i -lt $items.Count; $i++) {
    $t = $items[$i].Current.Name
    Write-Output ("  [{0}] {1}" -f $i, $t)
    # 判据要**窄**: 
    #   * 只认那一行的**开头形状** ("—— ⚠ 未载入任何模型权重") —— 用 -match "权重"
    #     会把别的窗口里的任意含"权重"两字的文本也匹配上 (实测踩过: 桌面上另一个
    #     chess.exe 的自检面板里就有这种句子, 于是最后一条被选成了它);
    #   * 位置不固定: 空的逐局明细会先放一行 "(还没有对局)" 占位, 我们的提示在它后面。
    if ($t -match "^\s*——\s*⚠\s*未载入任何模型权重") {
        if ($weightLine -eq "") { $weightLine = $t }
    }
}

$isWeightLine = ($weightLine -ne "")
Write-Output ("a list line is the weight notice = {0}" -f $isWeightLine)
if ($isWeightLine) {
    Write-Output ("weight notice line = {0}" -f $weightLine)
} else {
    Write-Output "NOTE: 没找到那一行。若桌面上还有**别的** chess.exe 在跑, 先关掉再跑本脚本"
}

try { $proc.CloseMainWindow() | Out-Null } catch { }
Start-Sleep -Seconds 2
if (-not $proc.HasExited) { $proc.Kill() }
Remove-Item $sandbox -Recurse -Force -ErrorAction SilentlyContinue

if ($isWeightLine) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }
