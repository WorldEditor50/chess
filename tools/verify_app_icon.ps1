# tools/verify_app_icon.ps1
#
# 校验程序图标 (src/app.ico / src/app.png) 以及"exe 真的带上了它"。
#
# 检查四件事:
#   1. src/app.ico 的容器结构: 条目数、尺寸、偏移、每个负载都是 PNG 且宽高对得上;
#   2. "帅"字形真的画出来了 (拿一个**必然缺字**的私用区码位做对照比墨量, 缺字会被
#      渲染成方框, 墨量分布和真字不一样);
#   3. 运行时 Qt 能加载它 (截图之外只有 app 自己知道; 这里读 chess.exe 的启动日志
#      "[icon] ..." 一行);
#   4. chess.exe 的文件图标就是我们的图标 (ExtractAssociatedIcon 后与 32x32 渲染比对)。
#
# ASCII only, no BOM (Windows PowerShell 会把无 BOM 的 .ps1 按 ANSI 解码)。
# 中文字形/窗口名一律用码位拼出来。

param(
    [string]$Exe = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$srcDir = (Resolve-Path (Join-Path $PSScriptRoot "..\src")).Path
$icoPath = Join-Path $srcDir "app.ico"
$pngPath = Join-Path $srcDir "app.png"

$checks = 0
$failed = 0
function Check([bool]$cond, [string]$msg) {
    $script:checks++
    if (-not $cond) { $script:failed++; Write-Output ("  [FAIL] " + $msg) }
}

# ---------------------------------------------------------------- 1. ICO 结构
Write-Output "[1] ICO container"
$bytes = [System.IO.File]::ReadAllBytes($icoPath)
$count = [System.BitConverter]::ToUInt16($bytes, 4)
Write-Output ("    file {0} bytes, {1} entries" -f $bytes.Length, $count)
Check -cond ($bytes.Length -gt 20000) -msg "文件大小像样 (不是空负载)"
Check -cond ([System.BitConverter]::ToUInt16($bytes, 0) -eq 0) -msg "reserved == 0"
Check -cond ([System.BitConverter]::ToUInt16($bytes, 2) -eq 1) -msg "type == 1 (icon)"
Check -cond ($count -ge 5) -msg "至少有 5 个尺寸"

$sizesSeen = @()
for ($i = 0; $i -lt $count; $i++) {
    $base = 6 + 16 * $i
    $w = [int]$bytes[$base]
    $h = [int]$bytes[$base + 1]
    $len = [System.BitConverter]::ToUInt32($bytes, $base + 8)
    $off = [System.BitConverter]::ToUInt32($bytes, $base + 12)
    if ($w -eq 0) { $w = 256 }
    if ($h -eq 0) { $h = 256 }
    $isPng = ($bytes[$off] -eq 0x89 -and $bytes[$off + 1] -eq 0x50 -and
              $bytes[$off + 2] -eq 0x4E -and $bytes[$off + 3] -eq 0x47)
    # PNG IHDR: 8 字节签名 + 4 字节长度 + "IHDR" + 4 字节宽 + 4 字节高 (都是大端)
    $pw = [int]$bytes[$off + 16] * 16777216 + [int]$bytes[$off + 17] * 65536 +
          [int]$bytes[$off + 18] * 256 + [int]$bytes[$off + 19]
    $ph = [int]$bytes[$off + 20] * 16777216 + [int]$bytes[$off + 21] * 65536 +
          [int]$bytes[$off + 22] * 256 + [int]$bytes[$off + 23]
    $sizesSeen += $w
    Write-Output ("    entry {0}: {1}x{2}, {3} bytes at {4}, png={5}, ihdr={6}x{7}" -f
                  $i, $w, $h, $len, $off, $isPng, $pw, $ph)
    Check -cond ($isPng) -msg "entry $i 是 PNG"
    Check -cond ($pw -eq $w -and $ph -eq $h) -msg "entry $i 的 PNG 宽高与目录一致"
    Check -cond ($off + $len -le $bytes.Length) -msg "entry $i 的偏移+长度在文件内"
}
Check(($sizesSeen -contains 16) -and ($sizesSeen -contains 32) -and
      ($sizesSeen -contains 256), "包含 16/32/256 三个关键尺寸")

# ---------------------------------------------------------------- 2. 字形
Write-Output "[2] glyph rendering"
function Get-Ink([string]$text, [System.Drawing.FontFamily]$fam, [int]$px) {
    $bmp = New-Object System.Drawing.Bitmap(64, 64)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.Clear([System.Drawing.Color]::White)
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias
    $font = New-Object System.Drawing.Font($fam, [float]$px,
                [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $fmt = New-Object System.Drawing.StringFormat
    $fmt.Alignment = [System.Drawing.StringAlignment]::Center
    $fmt.LineAlignment = [System.Drawing.StringAlignment]::Center
    $g.DrawString($text, $font, (New-Object System.Drawing.SolidBrush(
        [System.Drawing.Color]::Black)), (New-Object System.Drawing.RectangleF(0, 0, 64, 64)), $fmt)
    $g.Dispose()
    $ink = 0
    $rows = 0
    for ($y = 0; $y -lt 64; $y++) {
        $rowInk = 0
        for ($x = 0; $x -lt 64; $x++) {
            $p = $bmp.GetPixel($x, $y)
            if ($p.R -lt 128) { $ink++; $rowInk++ }
        }
        if ($rowInk -gt 0) { $rows++ }
    }
    $bmp.Dispose()
    return @($ink, $rows)
}

$fam = $null
foreach ($name in @("Microsoft YaHei", "SimHei", "SimSun", "NSimSun")) {
    try { $fam = New-Object System.Drawing.FontFamily($name); break } catch { }
}
$marshal = [string][char]0x5E05                       # U+5E05 = the red piece glyph
$missing = [string][char]0xE000                       # private use area: no font has it
$inkReal = Get-Ink $marshal $fam 44
$inkMissing = Get-Ink $missing $fam 44
Write-Output ("    font = {0}; ink(real) = {1} px over {2} rows; ink(missing) = {3} px" -f
              $fam.Name, $inkReal[0], $inkReal[1], $inkMissing[0])
Check -cond ($inkReal[0] -gt 60) -msg "真字形有足够的墨量 (不是空白)"
Check -cond ($inkReal[1] -ge 12) -msg "真字形有多行墨 (不是一条横线/方框)"
Check -cond ([math]::Abs($inkReal[0] - $inkMissing[0]) -gt 20) -msg (
      "真字形与缺字方框的墨量明显不同 (字体真的有这个字)")
# ---------------------------------------------------------------- 3. 运行时加载
Write-Output "[3] runtime load (from chess.exe startup log)"
if (Test-Path $Exe) {
    $env:PATH = (Split-Path (Get-ChildItem -Path "C:\Qt\*\msvc*\bin\Qt6Widgets.dll" |
                Select-Object -First 1).FullName -Parent) + ";" + $env:PATH
    $log = Join-Path $env:TEMP "icon_check_err.txt"
    $p = Start-Process -FilePath $Exe -WorkingDirectory (Split-Path $Exe -Parent) `
             -PassThru -RedirectStandardError $log
    Start-Sleep -Seconds 4
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    $lines = @(Get-Content $log -ErrorAction SilentlyContinue)
    $iconLine = $lines | Where-Object { $_ -like "*[icon]*" } | Select-Object -First 1
    Write-Output ("    " + $iconLine)
    Check -cond (($null -ne $iconLine) -and ($iconLine -like "*ok*")) -msg "app 启动时报告图标加载成功"
} else {
    Write-Output ("    (skip: {0} not built)" -f $Exe)
}

# ---------------------------------------------------------------- 4. exe 文件图标
Write-Output "[4] exe file icon"
if (Test-Path $Exe) {
    $exeIcon = [System.Drawing.Icon]::ExtractAssociatedIcon($Exe)
    Check -cond ($null -ne $exeIcon) -msg "exe 有可提取的图标"
    if ($null -ne $exeIcon) {
        $bmp = $exeIcon.ToBitmap()
        Write-Output ("    extracted {0}x{1}" -f $bmp.Width, $bmp.Height)
        # 我们的图标主色是红 (B3 28 20) 与木色 (E8 D5 A8); 默认的 exe 图标是蓝白窗口。
        $red = 0
        $wood = 0
        for ($y = 0; $y -lt $bmp.Height; $y++) {
            for ($x = 0; $x -lt $bmp.Width; $x++) {
                $c = $bmp.GetPixel($x, $y)
                if ($c.R -gt 120 -and $c.G -lt 90 -and $c.B -lt 90) { $red++ }
                if ($c.R -gt 200 -and $c.G -gt 180 -and $c.B -gt 130 -and $c.B -lt 210) { $wood++ }
            }
        }
        Write-Output ("    pixels: red-ish = {0}, wood-ish = {1} (of {2})" -f
                      $red, $wood, ($bmp.Width * $bmp.Height))
        Check -cond ($red -gt 10) -msg "exe 图标里有我们的红色棋子"
        Check -cond ($wood -gt 10) -msg "exe 图标里有我们的木色棋盘底"
        $bmp.Dispose()
    }
} else {
    Write-Output ("    (skip: {0} not built)" -f $Exe)
}

Write-Output ""
Write-Output ("=== {0} 项检查, {1} 项失败 ===" -f $checks, $failed)
if ($failed -eq 0) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }
