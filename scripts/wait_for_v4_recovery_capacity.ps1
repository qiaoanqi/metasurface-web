param(
    [int]$MinFreeMemoryGB = 16,
    [int]$PollSeconds = 60,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $root '.state'
$statusPath = Join-Path $stateDir 'reference_resolution_budget_v4_recovery_launcher.json'
$runner = Join-Path $root 'scripts\recover_reference_budget_v4_memory_failure.py'
$stdout = Join-Path $stateDir 'reference_budget_v4_recovery_stdout.log'
$stderr = Join-Path $stateDir 'reference_budget_v4_recovery_stderr.log'

function Write-AtomicStatus([hashtable]$Payload) {
    $Payload['observed_at'] = (Get-Date).ToString('o')
    $temporary = "$statusPath.$PID.tmp"
    $Payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding utf8
    [System.IO.File]::Move($temporary, $statusPath, $true)
}

while ($true) {
    $existing = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -like '*recover_reference_budget_v4_memory_failure.py*'
    }
    if ($existing) {
        Write-AtomicStatus @{
            status = 'already_running'
            recovery_pid = [int]$existing[0].ProcessId
        }
        exit 0
    }

    $game = Get-Process -Name 'EscapeFromTarkov' -ErrorAction SilentlyContinue
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $ready = (-not $game) -and ($freeGB -ge $MinFreeMemoryGB)
    if ($ready) {
        if ($CheckOnly) {
            Write-AtomicStatus @{
                status = 'ready'
                game_running = $false
                free_memory_gb = $freeGB
                minimum_free_memory_gb = $MinFreeMemoryGB
            }
            exit 0
        }
        $python = (Get-Command python).Source
        $process = Start-Process -FilePath $python `
            -ArgumentList @($runner, '--n-jobs', '16') `
            -WorkingDirectory $root `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru
        $process.PriorityClass = 'BelowNormal'
        Write-AtomicStatus @{
            status = 'launched'
            recovery_pid = $process.Id
            game_running = $false
            free_memory_gb = $freeGB
            minimum_free_memory_gb = $MinFreeMemoryGB
            n_jobs = 16
        }
        exit 0
    }

    Write-AtomicStatus @{
        status = 'waiting_for_capacity'
        game_running = [bool]$game
        game_pid = if ($game) { [int]$game.Id } else { $null }
        free_memory_gb = $freeGB
        minimum_free_memory_gb = $MinFreeMemoryGB
    }
    if ($CheckOnly) {
        exit 2
    }
    Start-Sleep -Seconds ([math]::Max(15, $PollSeconds))
}
