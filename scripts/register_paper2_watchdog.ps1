param(
    [switch]$Unregister,
    [switch]$StartNow
)

$ErrorActionPreference = 'Stop'
$taskName = 'Paper2-Unattended-Watchdog'
$projectRoot = Split-Path -Parent $PSScriptRoot
$watchdog = Join-Path $projectRoot 'scripts\paper2_watchdog.py'
$python = (Get-Command python -ErrorAction Stop).Source

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    exit 0
}

if (-not (Test-Path -LiteralPath $watchdog)) {
    throw "Watchdog script not found: $watchdog"
}

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "`"$watchdog`" --interval 15 --restart-delay 30 --max-restarts-per-hour 6" `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description 'Paper 2 local controller watchdog; no pool or paper mutation.' `
    -Force | Out-Null

if ($StartNow) {
    Start-ScheduledTask -TaskName $taskName
}

Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
