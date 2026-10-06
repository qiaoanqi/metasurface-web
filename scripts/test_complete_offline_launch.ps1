param([Parameter(Mandatory=$true)][string]$PackageRoot, [switch]$KeepRunning)
$ErrorActionPreference = 'Stop'
$PackageRoot = (Resolve-Path -LiteralPath $PackageRoot).Path
$manifest = Get-Content -LiteralPath (Join-Path $PackageRoot 'RELEASE_MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$startBat = Join-Path $PackageRoot $manifest.entrypoints.interactive
. (Join-Path $PackageRoot 'common_offline.ps1')
$results = [ordered]@{}
$listener = $null
$testState = $null
function Assert-That($Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Invoke-Cp936([string]$Script, [string]$Arguments = '') {
    & cmd.exe /d /c "chcp 936 >nul & call `"$Script`" $Arguments <nul"
    Assert-That ($LASTEXITCODE -eq 0) "CP936 invocation failed: $Script"
}
function Stop-TestInstance($State) {
    if (-not $State) { return }
    # Only clean up the exact host this verifier just launched. Its job handles
    # terminate the service; no user-facing stop script is needed.
    $owned = Get-OwnedProcess $State
    Assert-That ([bool]$owned) 'Test host identity changed; refusing to terminate it'
    Stop-Process -Id $owned.ProcessId -Force
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline -and
           (Get-Process -Id $State.server_process_id -ErrorAction SilentlyContinue)) {
        Start-Sleep -Milliseconds 100
    }
    Assert-That (-not (Get-Process -Id $State.server_process_id -ErrorAction SilentlyContinue)) 'Host exit left service running'
    Assert-That (-not (Test-RuntimeReady $State.port)) 'Host exit left the endpoint running'
    $current = Read-RuntimeState
    if ($current -and $current.process_id -eq $State.process_id) { Clear-RuntimeState }
    foreach ($name in @('streamlit.out.log','streamlit.err.log','desktop-host.lock')) {
        $handle = [IO.File]::Open((Join-Path $LogDir $name), 'Open', 'ReadWrite', 'None')
        $handle.Dispose()
    }
}
$bytes = [IO.File]::ReadAllBytes($startBat)
Assert-That (-not ($bytes | Where-Object { $_ -gt 127 })) 'Batch content must be ASCII without BOM'
Assert-That ([Text.Encoding]::ASCII.GetString($bytes).StartsWith("@echo off`r`n")) 'Expected CRLF batch entry'
$results.batch_ascii_crlf = $true
foreach ($ps1 in @('common_offline.ps1','start_offline.ps1')) {
    $tokens = $null; $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PackageRoot $ps1),[ref]$tokens,[ref]$parseErrors)
    Assert-That ($parseErrors.Count -eq 0) "PowerShell parse failure: $ps1"
}
$results.windows_powershell_parse = $true
Assert-That (-not (Get-OwnedProcess (Read-RuntimeState))) 'Package already running; verifier will not interrupt that instance'
try {
    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback,8512)
    $listener.Start()
    Invoke-Cp936 $startBat '-NoBrowser'
    $testState = Read-RuntimeState
    Assert-That ($testState.port -eq 8513) 'Busy port was not avoided'
    $results.busy_port = [int]$testState.port
    Invoke-Cp936 $startBat '-NoBrowser'
    $repeated = Read-RuntimeState
    Assert-That ($repeated.process_id -eq $testState.process_id -and $repeated.port -eq $testState.port) 'Duplicate start changed process or port'
    $results.duplicate_start = $true
    Stop-TestInstance $testState
    $testState = $null
    $results.host_exit_stops_service_and_releases_files = $true
    $listener.Stop(); $listener = $null

    # A stale state referring to this verifier must not terminate it on restart.
    $fake = [ordered]@{
        mode='desktop_window'; root=$PackageRoot; runtime=(Join-Path $PackageRoot 'runtime')
        executable=(Join-Path $PackageRoot '.venv\Scripts\python.exe')
        host_executable='invalid'; process_id=$PID; port=8512; created_at_utc='invalid'
    }
    $fake | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
    # ShellExecute uses the Windows .bat association, as Explorer does.
    $shellRun = Start-Process -FilePath $startBat -ArgumentList '-NoBrowser' -WorkingDirectory $PackageRoot -WindowStyle Hidden -PassThru
    Assert-That ($shellRun.WaitForExit(120000)) 'Launcher did not exit within two minutes'
    Assert-That ($shellRun.ExitCode -eq 0) 'Windows file-association launch failed'
    $testState = Read-RuntimeState
    Assert-That ([bool](Get-Process -Id $PID -ErrorAction SilentlyContinue)) 'Unrelated process was terminated'
    $results.recycled_pid_protection = $true
    Assert-That (Test-RuntimeReady $testState.port) 'Final service bootstrap failed'
    $config = Get-Content -LiteralPath (Join-Path $PackageRoot 'showcase\local-demo-config.js') -Raw
    Assert-That ($config.Contains("http://127.0.0.1:$($testState.port)/")) 'Showcase did not bind actual port'
    $results.shell_execute = $true
    $results.showcase_port_binding = $true
    $results.final_url = "http://127.0.0.1:$($testState.port)/"
    $results.final_process_id = $testState.process_id
    if (-not $KeepRunning) { Stop-TestInstance $testState; $testState = $null }
    $results.kept_running_for_browser_check = [bool]$KeepRunning
    $results | ConvertTo-Json
    $results | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $LogDir 'launcher-verification.json') -Encoding UTF8
} catch {
    if (-not $testState) { $testState = Read-RuntimeState }
    if ($testState -and $testState.process_id -ne $PID -and (Get-OwnedProcess $testState)) {
        Stop-TestInstance $testState
    }
    throw
} finally { if ($listener) { $listener.Stop() } }
