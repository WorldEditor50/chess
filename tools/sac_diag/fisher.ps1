param([int]$a, [int]$b, [int]$c, [int]$d)
# Two-sided Fisher exact test on the 2x2 table  [[a b],[c d]].
# a/c = wins, b/d = losses for the two arms (draws are excluded, as in the docs).
# ASCII-only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
function logFact([int]$n) {
    $s = 0.0
    for ($i = 2; $i -le $n; $i++) { $s += [math]::Log([double]$i) }
    return $s
}
function logChoose([int]$n, [int]$k) {
    if ($k -lt 0 -or $k -gt $n) { return [double]::NegativeInfinity }
    return (logFact $n) - (logFact $k) - (logFact ($n - $k))
}
$r1 = $a + $b
$r2 = $c + $d
$c1 = $a + $c
$n = $r1 + $r2
$logDen = logChoose $n $c1
$logP0 = (logChoose $r1 $a) + (logChoose $r2 $c) - $logDen
$p0 = [math]::Exp($logP0)
$sum = 0.0
for ($x = 0; $x -le $r1; $x++) {
    $y = $c1 - $x
    if ($y -lt 0 -or $y -gt $r2) { continue }
    $lp = (logChoose $r1 $x) + (logChoose $r2 $y) - $logDen
    if ($lp -le $logP0 + 1e-9) { $sum += [math]::Exp($lp) }
}
$or = 0.0
if ($b -gt 0 -and $c -gt 0) { $or = ([double]$a * $d) / ([double]$b * $c) }
Write-Output ("table [[{0} {1}][{2} {3}]]  p0={4:E3}  two-sided p={5:E3}  oddsRatio={6:N2}" -f $a, $b, $c, $d, $p0, $sum, $or)
