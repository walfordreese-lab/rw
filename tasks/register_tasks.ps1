$TasksDir = "C:\Users\reese\frd_backtest\tasks"

$tasks = @(
    @{ Name = "FRD_BreakoutScanner"; Xml = "breakout_scanner_task.xml";  Desc = "Daily breakout scanner (Mon-Fri 4:30 PM)" },
    @{ Name = "FRD_MomentumScanner"; Xml = "momentum_scanner_task.xml";  Desc = "Monthly momentum scanner (1st of month 4:30 PM)" },
    @{ Name = "FRD_PullbackScanner"; Xml = "pullback_scanner_task.xml";  Desc = "Daily pullback scanner (Mon-Fri 4:30 PM)" },
    @{ Name = "FRD_IntradayScanner"; Xml = "intraday_scanner_task.xml";  Desc = "Intraday scanner Strategy G (Mon-Fri 9:25 AM)" }
)

Write-Host "`nRegistering FRD scanner tasks...`n" -ForegroundColor Cyan

foreach ($t in $tasks) {
    $xmlPath = Join-Path $TasksDir $t.Xml
    if (-not (Test-Path $xmlPath)) {
        Write-Host "  [SKIP] $($t.Name) - XML not found" -ForegroundColor Yellow
        continue
    }
    $existing = Get-ScheduledTask -TaskName $t.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $t.Name -Confirm:$false
        Write-Host "  [REMOVED] existing: $($t.Name)" -ForegroundColor DarkGray
    }
    Register-ScheduledTask -TaskName $t.Name -Xml (Get-Content $xmlPath -Raw) -Force -ErrorAction SilentlyContinue | Out-Null
    $registered = Get-ScheduledTask -TaskName $t.Name -ErrorAction SilentlyContinue
    if ($registered) {
        Write-Host "  [OK] $($t.Name) - $($t.Desc)" -ForegroundColor Green
    } else {
        Write-Host "  [FAILED] $($t.Name)" -ForegroundColor Red
    }
}

Write-Host "`nRegistered FRD tasks:`n" -ForegroundColor Cyan
Get-ScheduledTask | Where-Object { $_.TaskName -like "FRD_*" } | Select-Object TaskName, State | Format-Table -AutoSize
