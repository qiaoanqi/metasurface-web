param([Parameter(Mandatory=$true)][string]$PackageRoot)
$ErrorActionPreference = 'Stop'
$PackageRoot = (Resolve-Path -LiteralPath $PackageRoot).Path
$manifest = Get-Content -LiteralPath (Join-Path $PackageRoot 'RELEASE_MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$startBat = Join-Path $PackageRoot $manifest.entrypoints.interactive
$stopBat = Join-Path $PackageRoot $manifest.entrypoints.stop
$statePath = Join-Path $PackageRoot 'logs\runtime-state.json'
$results = [ordered]@{}
$listener = $null
function Assert-That($Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Invoke-Cp936([string]$Script, [string]$Arguments = '') {
    & cmd.exe /d /c "chcp 936 >nul & call `"$Script`" $Arguments <nul"
    Assert-That ($LASTEXITCODE -eq 0) "CP936 invocation failed: $Script"
}
foreach ($bat in @($startBat,$stopBat)) {
    $bytes = [IO.File]::ReadAllBytes($bat)
    Assert-That (-not ($bytes | Where-Object { $_ -gt 127 })) 'Batch content must be ASCII without BOM'
    Assert-That ([Text.Encoding]::ASCII.GetString($bytes).StartsWith("@echo off`r`n")) 'Expected CRLF batch entry'
}
$results.batch_ascii_crlf = $true
foreach ($ps1 in @('common_offline.ps1','start_offline.ps1','stop_offline.ps1')) {
    $tokens = $null; $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PackageRoot $ps1),[ref]$tokens,[ref]$parseErrors)
    Assert-That ($parseErrors.Count -eq 0) "PowerShell parse failure: $ps1"
}
$results.windows_powershell_parse = $true
try {
    Invoke-Cp936 $stopBat
    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback,8512)
    $listener.Start()
    Invoke-Cp936 $startBat '-NoBrowser'
    $first = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-That ($first.port -eq 8513) 'Busy port was not avoided'
    $results.busy_port = [int]$first.port
    Invoke-Cp936 $startBat '-NoBrowser'
    $repeated = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-That ($repeated.process_id -eq $first.process_id -and $repeated.port -eq $first.port) 'Duplicate start changed process or port'
    $results.duplicate_start = $true
    Invoke-Cp936 $stopBat
    Assert-That (-not (Get-Process -Id $first.process_id -ErrorAction SilentlyContinue)) 'Stop left launcher running'
    Assert-That (-not (Test-Path -LiteralPath $statePath)) 'Stop left stale state'
    Invoke-Cp936 $stopBat
    $results.stop_and_repeated_stop = $true
    # Simulate a recycled PID belonging to this verifier, never to the app.
    $fake = [ordered]@{
        root=$PackageRoot; runtime=(Join-Path $PackageRoot 'runtime')
        executable=(Join-Path $PackageRoot '.venv\Scripts\python.exe')
        process_id=$PID; port=8513; created_at_utc='invalid'
    }
    $fake | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    Invoke-Cp936 $stopBat
    Assert-That ([bool](Get-Process -Id $PID -ErrorAction SilentlyContinue)) 'Unrelated process was terminated'
    $results.recycled_pid_protection = $true
} finally { if ($listener) { $listener.Stop() } }

# ShellExecute uses the Windows .bat association, as Explorer does.
$shellRun = Start-Process -FilePath $startBat -WorkingDirectory $PackageRoot -WindowStyle Hidden -PassThru
# Start-Process -Wait waits for the persistent service's entire process tree.
# Wait only for the entry-point window to exit, as a user would observe it.
Assert-That ($shellRun.WaitForExit(120000)) 'Launcher did not exit within two minutes'
Assert-That ($shellRun.ExitCode -eq 0) 'Windows file-association launch failed'
$final = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
$health = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$($final.port)/_stcore/health" -TimeoutSec 5
Assert-That ($health.StatusCode -eq 200 -and $health.Content.Trim() -eq 'ok') 'Final service is not healthy'
$config = Get-Content -LiteralPath (Join-Path $PackageRoot 'showcase\local-demo-config.js') -Raw
Assert-That ($config.Contains("http://127.0.0.1:$($final.port)/")) 'Showcase did not bind actual port'
$results.shell_execute = $true
$results.showcase_port_binding = $true
$results.final_url = "http://127.0.0.1:$($final.port)/"
$results.final_process_id = $final.process_id
$results | ConvertTo-Json
$results | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PackageRoot 'logs\launcher-verification.json') -Encoding UTF8
