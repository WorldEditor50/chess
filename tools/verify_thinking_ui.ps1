# tools/verify_thinking_ui.ps1
#
# Reproducible check for the "AI is thinking" feedback added in
# docs/issues_review.md section "零之二点八". Launches chess.exe, locates the
# board by its tan background, plays one move, and samples pixels to prove:
#
#   * the on-board status strip appears WHILE the AI thinks
#   * the ThinkingIndicator in the control column animates WHILE it thinks
#   * both are absent/static when idle and after the AI has answered
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File tools/verify_thinking_ui.ps1
#   powershell ... -File tools/verify_thinking_ui.ps1 -Exe <path to chess.exe> -KeepOpen
#
# Two pitfalls this script exists to encode:
#   1. The status strip is NOT drawn in its raw brush colour. QColor(28,38,54,230)
#      is alpha-composited over the tan board (222,184,135), landing near
#      (47,52,62). Looking for the raw colour finds nothing.
#   2. ABAgent answers in ~150 ms. Any trailing sleep after the click means the
#      first sample lands after thinking already finished, so sample immediately.
#
# ASCII only: Windows PowerShell reads .ps1 files as ANSI when they carry no BOM.

param(
    [string]$Exe = "",
    [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type -MemberDefinition '[DllImport("user32.dll")]public static extern void mouse_event(uint f,uint dx,uint dy,uint d,int e);' -Name MU -Namespace W32 -PassThru | Out-Null

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path
$exeDir = Split-Path $Exe -Parent

$grid = 60
$off = 50

function GrabBytes([int]$x, [int]$y, [int]$w, [int]$h) {
    $bmp = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($x, $y, 0, 0, (New-Object System.Drawing.Size($w, $h)))
    $g.Dispose()
    $rect = New-Object System.Drawing.Rectangle(0, 0, $w, $h)
    $d = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
                       [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $stride = $d.Stride
    $bytes = New-Object byte[] ($stride * $h)
    [System.Runtime.InteropServices.Marshal]::Copy($d.Scan0, $bytes, 0, $bytes.Length)
    $bmp.UnlockBits($d)
    $bmp.Dispose()
    return @{ bytes = $bytes; stride = $stride; w = $w; h = $h }
}

function CountDark($buf, [int]$lim) {
    $n = 0
    for ($yy = 0; $yy -lt $buf.h; $yy++) {
        $rowBase = $yy * $buf.stride
        for ($xx = 0; $xx -lt $buf.w; $xx++) {
            $i = $rowBase + $xx * 4
            if ($buf.bytes[$i] -lt $lim -and $buf.bytes[$i + 1] -lt $lim -and
                $buf.bytes[$i + 2] -lt $lim) { $n++ }
        }
    }
    return $n
}

function DiffCount($a, $b, [int]$thresh) {
    $n = 0
    for ($yy = 0; $yy -lt $a.h; $yy++) {
        $rowBase = $yy * $a.stride
        for ($xx = 0; $xx -lt $a.w; $xx++) {
            $i = $rowBase + $xx * 4
            $d = [Math]::Abs($a.bytes[$i] - $b.bytes[$i]) +
                 [Math]::Abs($a.bytes[$i + 1] - $b.bytes[$i + 1]) +
                 [Math]::Abs($a.bytes[$i + 2] - $b.bytes[$i + 2])
            if ($d -gt $thresh) { $n++ }
        }
    }
    return $n
}

# Board top-left = bounding box of the tan background (ChessBoard paints its
# whole widget with QColor(222,184,135)).
function FindBoard() {
    $vs = [System.Windows.Forms.SystemInformation]::VirtualScreen
    $bmp = New-Object System.Drawing.Bitmap($vs.Width, $vs.Height)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($vs.Left, $vs.Top, 0, 0, $bmp.Size)
    $g.Dispose()
    $rect = New-Object System.Drawing.Rectangle(0, 0, $bmp.Width, $bmp.Height)
    $d = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
                       [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $stride = $d.Stride
    $bytes = New-Object byte[] ($stride * $bmp.Height)
    [System.Runtime.InteropServices.Marshal]::Copy($d.Scan0, $bytes, 0, $bytes.Length)
    $bmp.UnlockBits($d)
    $h = $bmp.Height
    $w = $bmp.Width
    $bmp.Dispose()
    $minX = [int]::MaxValue; $minY = [int]::MaxValue
    for ($y = 0; $y -lt $h; $y++) {
        $rowBase = $y * $stride
        for ($x = 0; $x -lt $w; $x++) {
            $i = $rowBase + $x * 4
            if ($bytes[$i] -eq 135 -and $bytes[$i + 1] -eq 184 -and $bytes[$i + 2] -eq 222) {
                if ($x -lt $minX) { $minX = $x }
                if ($y -lt $minY) { $minY = $y }
            }
        }
    }
    if ($minX -eq [int]::MaxValue) { throw "chess board (tan background) not found on screen" }
    return @(($minX + $vs.Left), ($minY + $vs.Top))
}

function ClickAt([int]$x, [int]$y) {
    [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point($x, $y)
    Start-Sleep -Milliseconds 120
    [W32.MU]::mouse_event(0x0002, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 40
    [W32.MU]::mouse_event(0x0004, 0, 0, 0, 0)
}

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
Write-Output "launched chess.exe pid=$($proc.Id)"
Start-Sleep -Seconds 6

try {
    $origin = FindBoard
    $bx = $origin[0]
    $by = $origin[1]
    Write-Output "board origin on screen = ($bx,$by)"

    # Use a slow agent so the thinking window is wide: focus the combo box
    # (Tab from the first button) and step it to MCTS (800 simulations, ~1 s).
    $ws = New-Object -ComObject WScript.Shell
    [void]$ws.AppActivate($proc.Id)
    Start-Sleep -Milliseconds 500
    [System.Windows.Forms.SendKeys]::SendWait("{TAB}")
    Start-Sleep -Milliseconds 200
    [System.Windows.Forms.SendKeys]::SendWait("{DOWN}")
    Start-Sleep -Milliseconds 500

    # status strip occupies widget-local x 64..534, y 2..24; sample its right
    # end (text is left aligned) so each frame stays cheap.
    $scanX = $bx + 474; $scanY = $by + 6; $scanW = 56; $scanH = 13
    $ctrlX = $bx + 606; $ctrlY = $by - 9; $ctrlW = 196; $ctrlH = 640

    $idleStrip = CountDark (GrabBytes $scanX $scanY $scanW $scanH) 110
    $c1 = GrabBytes $ctrlX $ctrlY $ctrlW $ctrlH
    Start-Sleep -Milliseconds 250
    $c2 = GrabBytes $ctrlX $ctrlY $ctrlW $ctrlH
    $idleDiff = DiffCount $c1 $c2 24

    # red cannon (row 7, col 1) -> (row 7, col 4): a legal opening move
    ClickAt ($bx + $off + $grid * 1) ($by + $off + $grid * 7)
    Start-Sleep -Milliseconds 250
    ClickAt ($bx + $off + $grid * 4) ($by + $off + $grid * 7)

    $frames = 0; $hits = 0; $best = 0
    $ctrlPairs = 0; $ctrlHits = 0; $ctrlBest = 0
    $prevCtrl = $null
    for ($i = 0; $i -lt 90; $i++) {
        $n = CountDark (GrabBytes $scanX $scanY $scanW $scanH) 110
        $frames++
        if ($n -gt 40) { $hits++ }
        if ($n -gt $best) { $best = $n }
        if ($i % 3 -eq 0) {
            $cb = GrabBytes $ctrlX $ctrlY $ctrlW $ctrlH
            if ($null -ne $prevCtrl) {
                $dd = DiffCount $prevCtrl $cb 24
                $ctrlPairs++
                if ($dd -gt 200) { $ctrlHits++ }
                if ($dd -gt $ctrlBest) { $ctrlBest = $dd }
            }
            $prevCtrl = $cb
        }
    }

    Start-Sleep -Seconds 4
    $afterStrip = CountDark (GrabBytes $scanX $scanY $scanW $scanH) 110

    Write-Output ""
    Write-Output "idle_strip_dark_pixels   = $idleStrip   (expect 0)"
    Write-Output "idle_ctrl_diff_pixels    = $idleDiff    (expect 0: no idle animation)"
    Write-Output "thinking_strip_frames    = $hits / $frames   (expect > 0)"
    Write-Output "thinking_strip_max_dark  = $best / $($scanW * $scanH)"
    Write-Output "thinking_ctrl_animating  = $ctrlHits / $ctrlPairs  (expect > 0)"
    Write-Output "thinking_ctrl_max_diff   = $ctrlBest"
    Write-Output "after_strip_dark_pixels  = $afterStrip  (expect 0)"

    $ok = ($idleStrip -eq 0) -and ($idleDiff -eq 0) -and ($hits -gt 0) -and
          ($ctrlHits -gt 0) -and ($afterStrip -eq 0)
    Write-Output ""
    if ($ok) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }
}
finally {
    if (-not $KeepOpen) {
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    }
}
