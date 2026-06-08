# run_momentum_scanner.ps1
# Wrapper for momentum_scanner.py that ensures it runs on the correct
# business day — handles months where the 1st falls on a weekend or holiday.
# Task Scheduler calls this instead of python directly.

$Python  = "C:\Users\reese\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$Script  = "C:\Users\reese\frd_backtest\momentum_scanner.py"
$WorkDir = "C:\Users\reese\frd_backtest"

# Find the first business day of the current month
$today = Get-Date
$firstOfMonth = Get-Date -Year $today.Year -Month $today.Month -Day 1

$candidate = $firstOfMonth
while ($candidate.DayOfWeek -eq "Saturday" -or $candidate.DayOfWeek -eq "Sunday") {
    $candidate = $candidate.AddDays(1)
}
$firstBizDay = $candidate

# Only run if today IS the first business day of the month
# (allows re-running the task manually mid-month without double-firing)
if ($today.Date -eq $firstBizDay.Date) {
    $dateArg = $today.ToString("yyyy-MM-dd")
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Running momentum scanner for $dateArg (first biz day of month)"
    Set-Location $WorkDir
    & $Python $Script --date $dateArg
} else {
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Today ($($today.ToString('yyyy-MM-dd'))) is not the first business day of the month ($($firstBizDay.ToString('yyyy-MM-dd'))) -- skipping."
}
