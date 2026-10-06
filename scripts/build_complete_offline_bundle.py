"""Build a structured, one-click Windows offline delivery bundle.

The bundle keeps the audited web runtime intact while separating showcase,
deployment, audit, and protocol material into named directories.  The
runtime is launched from its own directory because app.py resolves models
and competition assets relative to __file__.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

if __package__:
    from .verify_complete_offline_bundle import validate_analysis_runtime
else:
    from verify_complete_offline_bundle import validate_analysis_runtime


PACKAGE_DIR_NAME = "AI超表面结构色智能设计系统_完整离线包_v1"
RUNTIME_REFRESH_FILES = (
    'app.py', 'ml_module.py', 'scripts/audit_forward_mlp_v8_sub_conversion.py',
    'data/fano_vs_fdtd_smallD.png',
)


README = """# AI 超表面结构色智能设计系统：完整离线交付包

## 运行

1. 双击 `启动离线演示.bat`。
2. 交互页在独立应用窗口中打开；如果 8512 已被占用，会自动选择后续可用端口。
3. 点击应用窗口右上角 × 即可退出，后台服务和日志文件会一起释放。

展示页保留在 `showcase/index.html`，可单独打开。应用窗口使用已安装的 Microsoft Edge 或 Google Chrome，不与日常浏览器共享运行进程。

需要先安装可用的 Python 3.10+。首次运行会创建本地 `.venv`，缺少网页依赖时安装 `runtime/requirements-web.txt`，安装阶段可能需要网络。依赖安装完成后，交互页本身在本地运行。当前电脑已初始化的环境会保留，后续启动可以复用。

## 目录

- `showcase/`：展示页及其图片资源。
- `runtime/`：完整交互运行源码、模型、参考库和 Streamlit 配置。应用必须从该目录启动。
- `deployment/`：Linux/ECS 部署脚本和 Nginx/Systemd 配置。
- `audit/`：已审核数据、审计记录、旧版发布清单和 UI 发布清单。
- `protocols/`：运行时模型转换协议和相关说明。
- `logs/`：启动脚本生成的运行日志。
- `RELEASE_MANIFEST.json`：本包文件哈希清单。

## 现场边界

本包用于网站展示和交互演示，不包含 Paper1/Paper2 控制面、训练集、holdout、active pool 或长时间 RCWA 任务。参考库按已审核的精确记录工作，不把候选结果表述为全局最优或实验真值。

如果启动失败，窗口会保留错误提示；详细记录在 `logs/launcher.log`、`logs/desktop-host.log`、`logs/streamlit.err.log` 和 `logs/streamlit.out.log`。`logs/runtime-state.json` 记录实际端口；重复启动会复用正在运行的实例。内部英文名 `.ps1` 和 `desktop_host.py` 是启动器配套文件，请保留。
"""


AUDIT_README = """# 证据材料索引

- `tio2_air_day_audit_20260930.json`：参考数据完整性审计。
- `tio2_air_day_color_audit_20260930.json`：颜色转换与数据一致性审计。
- `RELEASE_MANIFEST_v3.json`：上一版已验证网站包的文件清单。
- `ui-release-manifest.json`：当前 `runtime/` 中 UI 源码的哈希清单；科研控制面文件不随离线包分发。
- 模型转换证据所绑定的审计脚本保留在 `runtime/scripts/`，交互分析会核对它的版本。
- `11_本地离线演示说明.md`、`12_校赛本地交付清单.md`、`13_公开资源索引.md`：演示和交付说明，复制在 `docs/`。
"""


LOG_README = """此目录由启动脚本写入运行日志。首次解压时可以为空。
"""


OFFLINE_GUIDE = """# 本地离线演示说明

## Windows 一键运行

1. 完整解压压缩包，双击包根目录的 `启动离线演示.bat`。
2. 交互页会在独立应用窗口中打开。关闭这个窗口即可退出，后台服务和日志文件自动释放。
3. 展示页在 `showcase/index.html`，可单独打开；其中的本地交互链接会使用启动时选定的端口。

需要可用的 Python 3.10+ 和已安装的 Edge 或 Chrome。首次安装缺少的依赖时需要网络，已初始化的 `.venv` 会复用。请在解压后的目录运行，不要在压缩软件预览窗口中运行。

## 推荐体验路线

- 侧栏“加载已审核示例” → 预览 → 高保真参考对照。
- 逆设计：选择目标色 → 智能网格 → 应用候选 → 导出 JSON/CSV。
- 光谱、映射、图案：查看分析或生成图案，再导出结果。

参考库只匹配已审核的精确几何。搜索返回模型候选，不等同于实验验证；双柱模型缺失时使用明确标注的解析基线。

## Linux/macOS 手动运行

在包根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r runtime/requirements-web.txt
cd runtime
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8512
```

浏览器打开 `http://127.0.0.1:8512/`，终端按 Ctrl+C 结束。Windows 的关窗退出功能由 Windows 启动器提供。

## 文件与排错

`runtime/` 保留交互源码、模型和参考库，`deployment/` 是服务器部署脚本，`audit/` 是审计与证据，`protocols/` 是协议。文件完整性以包根 `RELEASE_MANIFEST.json` 为准，它不登记自身和运行时生成的环境、日志。

启动失败时保留窗口里的错误提示，并查看 `logs/launcher.log` 或 `logs/desktop-host.log`。应用重复启动会复用现有实例，端口冲突会自动避让。
"""


def refresh_ui_manifest(package_root: Path) -> None:
    """Bind the shipped UI manifest to runtime bytes, not the working tree."""
    path = package_root / 'audit' / 'ui-release-manifest.json'
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding='utf-8'))
    files = {}
    excluded = []
    for relative in previous['files']:
        source = package_root / 'runtime' / relative
        if source.is_file():
            files[relative] = 'sha256:' + sha256(source)
        else:
            excluded.append(relative)
    write_text(path, json.dumps({
        'schema': 'ui-release-manifest-v1', 'status': 'pass', 'errors': [],
        'root': 'runtime', 'scope': 'shipped_ui_sources_only',
        'excluded_from_offline_bundle': sorted(set(excluded + previous.get('excluded_from_offline_bundle', []))),
        'files': files,
    }, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


COMMON_PS1 = r'''$Root = (Resolve-Path $PSScriptRoot).Path
$Runtime = Join-Path $Root "runtime"
$LogDir = Join-Path $Root "logs"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$StatePath = Join-Path $LogDir "runtime-state.json"
$PidPath = Join-Path $LogDir "streamlit.pid"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Get-OwnedProcess($State) {
    if (-not $State -or $State.root -ne $Root -or $State.runtime -ne $Runtime -or
        $State.executable -ne $VenvPython -or [int]$State.port -lt 1024 -or
        [int]$State.port -gt 65535) { return $null }
    $owned = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$State.process_id)" -ErrorAction SilentlyContinue
    try {
        # PowerShell 7 may parse JSON timestamps as DateTime; 5.1 keeps strings.
        # Compare UTC instants instead of differently formatted timestamp text.
        $recordedCreation = if ($State.created_at_utc -is [datetime]) {
            $State.created_at_utc.ToUniversalTime()
        } else {
            [DateTimeOffset]::Parse($State.created_at_utc, [Globalization.CultureInfo]::InvariantCulture).UtcDateTime
        }
    } catch { return $null }
    if (-not $owned -or $owned.CreationDate.ToUniversalTime() -ne $recordedCreation) { return $null }
    if ($State.mode -eq 'desktop_window') {
        if (-not $owned -or $owned.ExecutablePath -ne $State.host_executable -or
            $owned.CommandLine.IndexOf((Join-Path $Root 'desktop_host.py'), [StringComparison]::OrdinalIgnoreCase) -lt 0) { return $null }
        return $owned
    }
    if (-not $owned -or $owned.ExecutablePath -ne $VenvPython -or
        $owned.CommandLine -notmatch '\s-m\s+streamlit\s+run\s+app\.py\s' -or
        $owned.CommandLine -notmatch "--server.port\s+$($State.port)(\s|$)") { return $null }
    return $owned
}

function Test-RuntimeReady([int]$ActivePort) {
    try {
        $health = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$ActivePort/_stcore/health" -TimeoutSec 2
        return ($health.StatusCode -eq 200 -and $health.Content.Trim() -eq 'ok')
    } catch { return $false }
}

function Read-RuntimeState {
    if (Test-Path -LiteralPath $StatePath) {
        try { return (Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json) } catch {}
    }
    return $null
}

function Clear-RuntimeState {
    Remove-Item -LiteralPath $StatePath, $PidPath -Force -ErrorAction SilentlyContinue
}

'''


START_PS1 = r'''param([ValidateRange(1024,65515)][int]$Port = 8512, [switch]$NoBrowser)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot 'common_offline.ps1')
$OutLog = Join-Path $LogDir "streamlit.out.log"
$ErrLog = Join-Path $LogDir "streamlit.err.log"
$lock = $null
$transcribing = $false
$newState = $null
$hostProcess = $null
try {
    $lock = [IO.File]::Open((Join-Path $LogDir 'launcher.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    Start-Transcript -LiteralPath (Join-Path $LogDir 'launcher.log') -Append | Out-Null
    $transcribing = $true
if (-not (Test-Path (Join-Path $Runtime "app.py"))) {
    throw "runtime\app.py not found. Keep the package directory intact."
}

    $oldState = Read-RuntimeState
    if (Get-OwnedProcess $oldState) {
        if (-not (Test-RuntimeReady $oldState.port)) { throw "应用没有响应。请关闭应用窗口后重新启动；日志在 $ErrLog。" }
        if ($oldState.mode -eq 'desktop_window') {
            Write-Host "应用窗口已在运行：http://127.0.0.1:$($oldState.port)/"
            exit 0
        }
        # Migrate a legacy detached service into the window-owned lifecycle.
        & taskkill.exe /PID $oldState.process_id /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) { throw '旧实例停止失败。' }
    }
    Clear-RuntimeState

function Test-PortBusy([int]$CandidatePort) {
    try {
        return (@(Get-NetTCPConnection -LocalPort $CandidatePort -State Listen -ErrorAction SilentlyContinue).Count -gt 0)
    } catch {
        return $false
    }
}

$requestedPort = $Port
if (Test-PortBusy $Port) {
    for ($candidatePort = $Port + 1; $candidatePort -le $Port + 20; $candidatePort++) {
        if (-not (Test-PortBusy $candidatePort)) {
            $Port = $candidatePort
            break
        }
    }
    if ($Port -eq $requestedPort) {
        throw "端口 $requestedPort 已被占用，且后续 20 个端口也不可用。"
    }
    Write-Host "端口 $requestedPort 已被占用，交互页改用 http://127.0.0.1:$Port/。"
}

$baseExe = $null
function Test-PythonExecutable([string]$Candidate) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return $false }
    try {
        $probe = (& $Candidate -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null).Trim()
        if ($LASTEXITCODE -ne 0) { return $false }
        $probeParts = $probe -split '\.'
        return ($probeParts.Count -ge 2 -and [int]$probeParts[0] -ge 3 -and ([int]$probeParts[0] -gt 3 -or [int]$probeParts[1] -ge 10))
    } catch {
        return $false
    }
}

# Prefer a directly callable Python executable.  The Python launcher may have
# a broken default registration, so every candidate is probed before use.
if (Test-PythonExecutable $VenvPython) { $baseExe = $VenvPython }
$pythonCommands = @(Get-Command python -All -ErrorAction SilentlyContinue)
foreach ($command in $pythonCommands) {
    if ($baseExe) { break }
    if (Test-PythonExecutable $command.Source) { $baseExe = $command.Source; break }
}
if (-not $baseExe) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $registered = & $pyLauncher.Source --list-paths 2>$null
        foreach ($line in $registered) {
            if ($line -match '^\s*-V:\S+\s+(.+)$' -and (Test-PythonExecutable $Matches[1].Trim())) {
                $baseExe = $Matches[1].Trim()
                break
            }
        }
    }
}
if (-not $baseExe) {
    throw "未找到可运行的 Python 3.10+。请先安装 Python，再重新双击启动脚本。"
}

if (-not (Test-PythonExecutable $VenvPython)) {
    Write-Host "正在创建本地运行环境..."
    & $baseExe -m venv --system-site-packages (Join-Path $Root '.venv')
    if ($LASTEXITCODE -ne 0) { throw "创建 .venv 失败。" }
}

$dependencyProbe = & $VenvPython -c "import streamlit, numpy, onnxruntime, PIL, matplotlib, scipy"
if ($LASTEXITCODE -ne 0) {
    Write-Host "正在安装网页运行依赖；若电脑离线，请先准备 requirements-web.txt 对应的本地依赖。"
    & $VenvPython -m pip install -r (Join-Path $Runtime "requirements-web.txt")
    if ($LASTEXITCODE -ne 0) { throw "依赖安装失败。详情见终端输出。" }
}

# Verify imports and the audited reference library before announcing readiness.
Push-Location $Runtime
try {
    & $VenvPython -c "import engine, ml_module, rl_design; from competition.reference_library import load_reference_library; library = load_reference_library(); print('Reference library loaded')"
    if ($LASTEXITCODE -ne 0) { throw "运行文件检查失败，请查看 logs\launcher.log。" }
} finally { Pop-Location }

Write-Host "正在打开应用窗口..."
$hostArguments = @(('"' + (Join-Path $Root 'desktop_host.py') + '"'), '--port', ([string]$Port))
if ($NoBrowser) { $hostArguments += '--no-browser' }
$windowlessPython = Join-Path $Root '.venv\Scripts\pythonw.exe'
$hostProcess = Start-Process -FilePath $windowlessPython -ArgumentList $hostArguments -WorkingDirectory $Root -WindowStyle Hidden -PassThru
$ready = $false
$deadline = (Get-Date).AddSeconds(100)
while ((Get-Date) -lt $deadline) {
    $newState = Read-RuntimeState
    if ($newState -and $newState.status -eq 'ready' -and (Get-OwnedProcess $newState) -and (Test-RuntimeReady $newState.port)) {
        $ready = $true
        break
    }
    if ($hostProcess.HasExited) { throw "应用启动失败，请查看 logs\desktop-host.log。" }
    Start-Sleep -Milliseconds 500
}
if (-not $ready) { throw "应用未在 100 秒内就绪，请查看 logs\desktop-host.log。" }
Set-Content -LiteralPath $PidPath -Value ([string]$newState.process_id) -Encoding ascii
Write-Host "应用已打开：http://127.0.0.1:$($newState.port)/；关闭应用窗口即可退出。"
exit 0
} catch {
    if ($newState -and (Get-OwnedProcess $newState)) {
        & taskkill.exe /PID $newState.process_id /T /F | Out-Null
        Clear-RuntimeState
    } elseif ($hostProcess -and -not $hostProcess.HasExited) {
        & taskkill.exe /PID $hostProcess.Id /T /F | Out-Null
    }
    Write-Host "启动失败：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "请保留此窗口。日志目录：$LogDir"
    exit 1
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    if ($lock) { $lock.Dispose() }
}
'''


START_BAT = r'''@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_offline.ps1" %*
set "launcher_exit=%errorlevel%"
if not "%launcher_exit%"=="0" (
    echo Startup failed. See logs\launcher.log in this folder.
    pause
)
endlocal & exit /b %launcher_exit%
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, target, dirs_exist_ok=True)


def write_text(path: Path, text: str, *, bom: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")


def write_batch(path: Path, text: str) -> None:
    # cmd.exe does not reliably accept UTF-8 BOM or non-ASCII source under CP936.
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))


def write_launchers(project_root: Path, package_root: Path) -> None:
    # These exact legacy entry points were removed at the user's request.
    for name in ("停止离线演示.bat", "stop_offline.ps1"):
        (package_root / name).unlink(missing_ok=True)
    write_text(package_root / "README_先看这里.md", README)
    write_text(package_root / 'docs' / '11_本地离线演示说明.md', OFFLINE_GUIDE)
    write_text(package_root / 'audit' / 'README_证据索引.md', AUDIT_README)
    write_text(package_root / "common_offline.ps1", COMMON_PS1, bom=True)
    write_text(package_root / "start_offline.ps1", START_PS1, bom=True)
    copy_file(project_root / 'scripts' / 'offline_desktop_host.py', package_root / 'desktop_host.py')
    write_batch(package_root / "启动离线演示.bat", START_BAT)


def refresh_launchers(project_root: Path, output_zip: Path, staging: Path, *, refresh_runtime=False) -> dict[str, object]:
    """Refresh approved runtime files, validate analysis dependencies, then reseal."""
    manifest_path = staging / 'RELEASE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    write_launchers(project_root, staging)
    if refresh_runtime:
        for name in RUNTIME_REFRESH_FILES:
            copy_file(project_root / name, staging / 'runtime' / name)
    manifest['analysis_runtime'] = validate_analysis_runtime(staging / 'runtime')
    refresh_ui_manifest(staging)
    manifest['launcher_mode'] = 'desktop-window-close-stops-service'
    manifest['entrypoints'].pop('stop', None)
    files = {
        path.relative_to(staging).as_posix(): 'sha256:' + sha256(path)
        for path in sorted(staging.rglob('*'))
        if path.is_file() and path != manifest_path
        and '.venv' not in path.relative_to(staging).parts
        and '__pycache__' not in path.relative_to(staging).parts
        and path.relative_to(staging).as_posix() != 'showcase/local-demo-config.js'
        and (path.parent != staging / 'logs' or path.name == 'README.md')
    }
    manifest['files'] = files
    manifest['file_count'] = len(files)
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    temporary_zip = output_zip.with_suffix('.tmp.zip')
    with ZipFile(temporary_zip, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in sorted([*files, 'RELEASE_MANIFEST.json']):
            info = ZipInfo(f'{PACKAGE_DIR_NAME}/{relative}', date_time=(2020,1,1,0,0,0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, (staging / relative).read_bytes(), compress_type=ZIP_DEFLATED)
    temporary_zip.replace(output_zip)
    return manifest


def build(project_root: Path, source_package: Path, output_zip: Path, staging: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    source_package = source_package.resolve()
    output_zip = output_zip.resolve()
    staging = staging.resolve()
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    if output_zip.exists():
        raise FileExistsError(f"output ZIP already exists: {output_zip}")
    if not source_package.is_dir():
        raise FileNotFoundError(source_package)

    staging.mkdir(parents=True)
    runtime = staging / "runtime"
    for source in sorted(source_package.glob("*.py")):
        copy_file(source, runtime / source.name)
    for name in RUNTIME_REFRESH_FILES:
        copy_file(project_root / name, runtime / name)
    copy_file(source_package / "requirements-web.txt", runtime / "requirements-web.txt")
    copy_tree(source_package / ".streamlit", runtime / ".streamlit")
    copy_tree(source_package / "models", runtime / "models")
    copy_file(source_package / "competition" / "reference_library.py", runtime / "competition" / "reference_library.py")
    copy_file(source_package / "competition" / "tio2_air_reference_records_v1.jsonl", runtime / "competition" / "tio2_air_reference_records_v1.jsonl")

    copy_tree(source_package / "static", staging / "showcase")
    copy_tree(source_package / "deployment", staging / "deployment")
    copy_tree(source_package / "protocols", staging / "protocols")
    copy_tree(source_package / "protocols", runtime / "protocols")
    showcase_index = staging / "showcase" / "index.html"
    html = showcase_index.read_text(encoding="utf-8")
    html = html.replace("</head>", '<script src="local-demo-config.js"></script>\n</head>', 1)
    html = html.replace("localDemoHost ? 'http://127.0.0.1:8512' : '/app/'", "localDemoHost ? (window.OFFLINE_APP_URL || 'http://127.0.0.1:8512/') : '/app/'")
    write_text(showcase_index, html)

    docs = staging / "docs"
    audit = staging / "audit"
    for name in ("11_本地离线演示说明.md", "12_校赛本地交付清单.md", "13_公开资源索引.md"):
        copy_file(source_package / "competition" / name, docs / name)
    for name in ("tio2_air_day_audit_20260930.json", "tio2_air_day_color_audit_20260930.json"):
        copy_file(source_package / "competition" / name, audit / name)
        copy_file(source_package / "competition" / name, runtime / "competition" / name)
    copy_file(source_package / "RELEASE_MANIFEST.json", audit / "RELEASE_MANIFEST_v3.json")
    ui_manifest = project_root / "dist" / "ui-release-manifest.json"
    if ui_manifest.is_file():
        copy_file(ui_manifest, audit / "ui-release-manifest.json")
    write_text(audit / "README_证据索引.md", AUDIT_README)

    write_launchers(project_root, staging)
    analysis_runtime = validate_analysis_runtime(runtime)
    refresh_ui_manifest(staging)
    write_text(staging / "logs" / "README.md", LOG_README)

    files = {
        path.relative_to(staging).as_posix(): "sha256:" + sha256(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    source_zip = (
        project_root
        / "_archive"
        / "legacy_web_releases_20261006"
        / "ai_metasurface_web_bundle_v3_local_20261005.zip"
    )
    manifest = {
        "schema": "ai-metasurface-complete-offline-bundle-v1",
        "status": "pass",
        "analysis_runtime": analysis_runtime,
        "entrypoints": {
            "showcase": "showcase/index.html",
            "interactive": "启动离线演示.bat",
        },
        "source_web_bundle": {
            "path": str(source_package),
            "sha256": sha256(source_zip)
            if source_zip.is_file()
            else None,
        },
        "file_count": len(files),
        "generated_exclusions": [".venv/**", "logs/*", "showcase/local-demo-config.js"],
        "files": files,
    }
    write_text(staging / "RELEASE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(staging).as_posix()
            info = ZipInfo(f"{PACKAGE_DIR_NAME}/{relative}", date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED)
    return manifest


def sync_installed_package(staging: Path, installed: Path, manifest: dict) -> None:
    """Update release files while preserving the installed environment/logs."""
    installed = installed.resolve()
    if installed == staging.resolve() or installed.name != PACKAGE_DIR_NAME:
        raise ValueError('Expected a separate installed complete-offline package directory')
    prior = json.loads((installed / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    # Validate all overlaps first so local user edits cannot be overwritten.
    for relative, expected in manifest['files'].items():
        target = installed / relative
        target.resolve().relative_to(installed)
        if target.is_file():
            actual = 'sha256:' + sha256(target)
            if actual not in {expected, prior['files'].get(relative)}:
                raise ValueError(f'Installed file has user changes: {relative}')
    for relative, expected in manifest['files'].items():
        target = installed / relative
        if not target.is_file() or 'sha256:' + sha256(target) != expected:
            copy_file(staging / relative, target)
    for name in ('停止离线演示.bat', 'stop_offline.ps1'):
        (installed / name).unlink(missing_ok=True)
    copy_file(staging / 'RELEASE_MANIFEST.json', installed / 'RELEASE_MANIFEST.json')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--refresh-launchers', action='store_true', help='Update the existing staging launchers and atomically reseal its ZIP')
    parser.add_argument('--refresh-runtime', action='store_true', help='Also sync the current ML and candidate-display fixes when refreshing an existing bundle')
    parser.add_argument('--installed', type=Path, help='Sync an existing installed package; preserve .venv and logs')
    parser.add_argument(
        "--source-package",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "_archive"
        / "legacy_web_releases_20261006"
        / "AI超表面结构色智能设计系统_发布包_v3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dist" / "ai_metasurface_complete_offline_bundle_v1.zip",
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dist" / PACKAGE_DIR_NAME,
    )
    args = parser.parse_args()
    if args.refresh_runtime and not args.refresh_launchers:
        parser.error('--refresh-runtime requires --refresh-launchers')
    manifest = (refresh_launchers(args.project_root.resolve(), args.output.resolve(), args.staging.resolve(), refresh_runtime=args.refresh_runtime)
                if args.refresh_launchers else build(args.project_root, args.source_package, args.output, args.staging))
    if args.installed:
        sync_installed_package(args.staging.resolve(), args.installed, manifest)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "staging": str(args.staging.resolve()),
        "bytes": args.output.resolve().stat().st_size,
        "files": manifest["file_count"],
        "sha256": sha256(args.output.resolve()),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
