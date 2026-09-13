# tools/make_app_icon.ps1
#
# 生成程序图标 (src/app.png + src/app.ico)。
#
# 为什么用脚本生成而不是塞一个二进制进来:
#   * 图标改了要能重做 —— 二进制资源在 diff 里什么也看不出来, 也没法审;
#   * 需要的图形很简单 (棋盘底 + 一枚红"帅"棋子), 用 System.Drawing 画十几行就够,
#     不需要引入设计工具/额外依赖。
#
# 生成物:
#   src/app.png   256x256 (Qt 运行时用; 走 res.qrc 嵌进 exe)
#   src/app.ico   多尺寸 (16/24/32/48/64/128/256), Windows 可执行文件图标 (src/app.rc)
#
# 用法: powershell -ExecutionPolicy Bypass -File tools/make_app_icon.ps1
#
# 注: ICO 容器是手写的 (ICONDIR + ICONDIRENTRY + PNG 负载)。Vista 以后 Windows 允许
# 图标条目直接放 PNG, Qt 也一样支持 —— 这样就不必手搓 DIB/AND 掩码。

param(
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

if ($OutDir -eq "") {
    $OutDir = Join-Path $PSScriptRoot "..\src"
}
$OutDir = (Resolve-Path $OutDir).Path

# 配色 (与棋盘/界面一致: 米黄底 + 红方棋子)
$wood      = [System.Drawing.Color]::FromArgb(232, 213, 168)
$woodEdge  = [System.Drawing.Color]::FromArgb(169, 130, 90)
$gridLine  = [System.Drawing.Color]::FromArgb(201, 162, 107)
$pieceFill = [System.Drawing.Color]::FromArgb(247, 237, 216)
$pieceRed  = [System.Drawing.Color]::FromArgb(179, 40, 32)

function New-RoundedPath([int]$size, [int]$radius) {
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = $radius * 2
    $p.AddArc(0, 0, $d, $d, 180, 90)
    $p.AddArc($size - $d, 0, $d, $d, 270, 90)
    $p.AddArc($size - $d, $size - $d, $d, $d, 0, 90)
    $p.AddArc(0, $size - $d, $d, $d, 90, 90)
    $p.CloseFigure()
    return $p
}

# 找一个有中文字形的字体 (画"帅"字用)
function Get-CjkFontFamily() {
    foreach ($name in @("Microsoft YaHei", "SimHei", "SimSun", "NSimSun",
                        "Microsoft JhengHei", "Arial Unicode MS")) {
        try {
            $f = New-Object System.Drawing.FontFamily($name)
            return $f
        } catch {
            continue
        }
    }
    return (New-Object System.Drawing.FontFamily("Arial"))
}

function New-IconBitmap([int]$size) {
    $bmp = New-Object System.Drawing.Bitmap($size, $size,
        [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.Clear([System.Drawing.Color]::Transparent)

    $s = [double]$size

    # --- 棋盘底 (圆角方块 + 边框) ---
    $radius = [int]([math]::Max(2, [math]::Round($s * 0.16)))
    $path = New-RoundedPath $size $radius
    $g.FillPath((New-Object System.Drawing.SolidBrush($wood)), $path)
    $penEdge = New-Object System.Drawing.Pen($woodEdge, [float]([math]::Max(1.0, $s * 0.02)))
    $g.DrawPath($penEdge, $path)

    # --- 棋盘格线 (只画内部几条, 小尺寸下也看得清) ---
    $penGrid = New-Object System.Drawing.Pen($gridLine, [float]([math]::Max(1.0, $s * 0.012)))
    $m = $s * 0.14            # 边距
    $span = $s - 2 * $m
    for ($i = 1; $i -le 3; $i++) {
        $t = $m + $span * $i / 4.0
        $g.DrawLine($penGrid, [float]$t, [float]$m, [float]$t, [float]($s - $m))
        $g.DrawLine($penGrid, [float]$m, [float]$t, [float]($s - $m), [float]$t)
    }

    # --- 棋子 (外圈红环 + 内圈 + 米色底 + "帅") ---
    $cx = $s / 2.0
    $cy = $s / 2.0
    $rOuter = $s * 0.34
    $ringW = [float]([math]::Max(1.2, $s * 0.045))

    $g.FillEllipse((New-Object System.Drawing.SolidBrush($pieceFill)),
                   [float]($cx - $rOuter), [float]($cy - $rOuter),
                   [float]($rOuter * 2), [float]($rOuter * 2))
    $penRing = New-Object System.Drawing.Pen($pieceRed, $ringW)
    $g.DrawEllipse($penRing,
                   [float]($cx - $rOuter), [float]($cy - $rOuter),
                   [float]($rOuter * 2), [float]($rOuter * 2))
    $rInner = $rOuter * 0.80
    $penInner = New-Object System.Drawing.Pen($pieceRed, [float]([math]::Max(0.8, $s * 0.015)))
    $g.DrawEllipse($penInner,
                   [float]($cx - $rInner), [float]($cy - $rInner),
                   [float]($rInner * 2), [float]($rInner * 2))

    # 字: 小尺寸直接跳过 (16x16 画字只会糊成一团, 圆圈本身已经能认出来)
    if ($size -ge 24) {
        $fontSize = [float]($s * 0.40)
        $font = New-Object System.Drawing.Font((Get-CjkFontFamily), $fontSize,
                                               [System.Drawing.FontStyle]::Bold,
                                               [System.Drawing.GraphicsUnit]::Pixel)
        $fmt = New-Object System.Drawing.StringFormat
        $fmt.Alignment = [System.Drawing.StringAlignment]::Center
        $fmt.LineAlignment = [System.Drawing.StringAlignment]::Center
        $rect = New-Object System.Drawing.RectangleF(0, 0, [float]$s, [float]$s)
        $brush = New-Object System.Drawing.SolidBrush($pieceRed)
        $glyph = [string][char]0x5E05        # U+5E05 = the red marshal piece
        $g.DrawString($glyph, $font, $brush, $rect, $fmt)
    }

    $g.Dispose()
    return $bmp
}

function Get-PngBytes([System.Drawing.Bitmap]$bmp) {
    $ms = New-Object System.IO.MemoryStream
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $bytes = $ms.ToArray()
    $ms.Dispose()
    # `return ,$bytes` 的逗号很关键: PowerShell 会把 byte[] 拆成一个个字节输出, 不包一层
    # 的话调用方拿到的是"一堆字节"而不是"一个 byte[]" -- 于是 $pngs[$i].Length 变成 1,
    # 写出来的 ICO 每个条目只有 1 字节 (第一版就是这样: 整个 .ico 只有 125 字节)。
    return ,$bytes
}

$sizes = @(16, 24, 32, 48, 64, 128, 256)

# --- app.png (256, 运行时用) ---
$big = New-IconBitmap 256
$big.Save((Join-Path $OutDir "app.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$big.Dispose()
Write-Output ("wrote {0}" -f (Join-Path $OutDir "app.png"))

# --- app.ico (多尺寸, 每个条目都是 PNG) ---
# 注意用 ArrayList.Add 而不是 `+=`: PowerShell 的数组 `+=` 会把右边的 Byte[] **拆开**
# 逐个追加, 于是 $pngs 变成"一堆字节"而不是"每个尺寸一个 Byte[]" (第一版就是这么错的,
# 写出来的 .ico 每个条目长度都是 1)。
$pngList = New-Object System.Collections.ArrayList
foreach ($sz in $sizes) {
    $b = New-IconBitmap $sz
    [void]$pngList.Add((Get-PngBytes $b))
    $b.Dispose()
}
$pngs = $pngList.ToArray()

$ms = New-Object System.IO.MemoryStream
$bw = New-Object System.IO.BinaryWriter($ms)
$bw.Write([uint16]0)                  # reserved
$bw.Write([uint16]1)                  # type = icon
$bw.Write([uint16]$sizes.Count)       # image count
$offset = 6 + 16 * $sizes.Count
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $sz = $sizes[$i]
    $bw.Write([byte]($(if ($sz -ge 256) { 0 } else { $sz })))   # width (0 = 256)
    $bw.Write([byte]($(if ($sz -ge 256) { 0 } else { $sz })))   # height
    $bw.Write([byte]0)                # palette size
    $bw.Write([byte]0)                # reserved
    $bw.Write([uint16]1)              # color planes
    $bw.Write([uint16]32)             # bits per pixel
    $bw.Write([uint32]$pngs[$i].Length)
    $bw.Write([uint32]$offset)
    $offset += $pngs[$i].Length
}
foreach ($png in $pngs) { $bw.Write($png) }
$bw.Flush()
$icoPath = Join-Path $OutDir "app.ico"
[System.IO.File]::WriteAllBytes($icoPath, $ms.ToArray())
$bw.Dispose()
$ms.Dispose()
Write-Output ("wrote {0} ({1} sizes, {2} bytes)" -f $icoPath, $sizes.Count,
              (Get-Item $icoPath).Length)
