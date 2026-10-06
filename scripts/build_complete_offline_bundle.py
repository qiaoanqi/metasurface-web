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


PACKAGE_DIR_NAME = "AI超表面结构色智能设计系统_完整离线包_v1"


README = """# AI 超表面结构色智能设计系统：完整离线交付包

## 运行

1. 双击 `启动离线演示.bat`。
2. 脚本会启动展示页，并在浏览器打开本地交互页 `http://127.0.0.1:8512/`；如果 8512 已被占用，会自动选择后续可用端口并在窗口中提示实际地址。
3. 使用结束后双击 `停止离线演示.bat`。

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

如果启动失败，窗口会保留错误提示；详细记录在 `logs/launcher.log`、`logs/streamlit.err.log` 和 `logs/streamlit.out.log`。`logs/runtime-state.json` 记录实际端口；重复启动会打开这个地址。内部英文名 `.ps1` 是启动器配套文件，请保留。
"""


AUDIT_README = """# 证据材料索引

- `tio2_air_day_audit_20260930.json`：参考数据完整性审计。
- `tio2_air_day_color_audit_20260930.json`：颜色转换与数据一致性审计。
- `RELEASE_MANIFEST_v3.json`：上一版已验证网站包的文件清单。
- `ui-release-manifest.json`：UI 发布门禁清单（若随项目发布）。
- `11_本地离线演示说明.md`、`12_校赛本地交付清单.md`、`13_公开资源索引.md`：演示和交付说明，复制在 `docs/`。
"""


LOG_README = """此目录由启动脚本写入运行日志。首次解压时可以为空。
"""


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
    if (-not $owned -or $owned.ExecutablePath -ne $VenvPython -or
        $owned.CreationDate.ToUniversalTime().ToString('o') -ne $State.created_at_utc -or
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

function Open-OfflinePages([int]$ActivePort) {
    # This generated config is intentionally mutable and excluded from release hashes.
    $url = "http://127.0.0.1:$ActivePort/"
    Set-Content -LiteralPath (Join-Path $Root 'showcase\local-demo-config.js') -Encoding ASCII -Value "window.OFFLINE_APP_URL = '$url';"
    Start-Process (Join-Path $Root "showcase\index.html")
    Start-Process $url
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
try {
    $lock = [IO.File]::Open((Join-Path $LogDir 'launcher.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    Start-Transcript -LiteralPath (Join-Path $LogDir 'launcher.log') -Append | Out-Null
    $transcribing = $true
if (-not (Test-Path (Join-Path $Runtime "app.py"))) {
    throw "runtime\app.py not found. Keep the package directory intact."
}

    $oldState = Read-RuntimeState
    if (Get-OwnedProcess $oldState) {
        if (-not (Test-RuntimeReady $oldState.port)) { throw "已记录的交互进程没有响应。请先运行停止脚本，再启动；日志在 $ErrLog。" }
        if (-not $NoBrowser) { Open-OfflinePages $oldState.port }
        Write-Host "交互页已在运行：http://127.0.0.1:$($oldState.port)/"
        exit 0
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

Write-Host "正在启动交互页..."
$arguments = @(
    "-m", "streamlit", "run", "app.py",
    "--server.address", "127.0.0.1",
    "--server.port", ([string]$Port),
    "--server.headless", "true"
)
$process = Start-Process -FilePath $VenvPython -ArgumentList $arguments -WorkingDirectory $Runtime -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -WindowStyle Hidden -PassThru
$identity = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.Id)"
if (-not $identity) { throw "服务进程提前退出，请查看 $ErrLog。" }
$newState = [ordered]@{
    root = $Root; runtime = $Runtime; executable = $VenvPython
    process_id = $process.Id; port = $Port
    created_at_utc = $identity.CreationDate.ToUniversalTime().ToString('o')
}
$tempState = "$StatePath.tmp"
$newState | ConvertTo-Json | Set-Content -LiteralPath $tempState -Encoding UTF8
Move-Item -LiteralPath $tempState -Destination $StatePath -Force
Set-Content -LiteralPath $PidPath -Value ([string]$process.Id) -Encoding ascii
$ready = $false
$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline) {
    if (-not (Get-OwnedProcess ([pscustomobject]$newState))) { throw "服务进程提前退出，请查看 $ErrLog。" }
    if (Test-RuntimeReady $Port) { $ready = $true; break }
    Start-Sleep -Milliseconds 500
}
if (-not $ready) { throw "服务未在 90 秒内就绪，请查看 $ErrLog。" }
if (-not $NoBrowser) { Open-OfflinePages $Port }
Write-Host "本地服务已就绪：http://127.0.0.1:$Port/；停止时双击 停止离线演示.bat。"
exit 0
} catch {
    if ($newState -and (Get-OwnedProcess ([pscustomobject]$newState))) {
        & taskkill.exe /PID $newState.process_id /T /F | Out-Null
        Clear-RuntimeState
    }
    Write-Host "启动失败：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "请保留此窗口。日志目录：$LogDir"
    exit 1
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    if ($lock) { $lock.Dispose() }
}
'''


STOP_PS1 = r'''$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot 'common_offline.ps1')
$lock = $null
try {
    $lock = [IO.File]::Open((Join-Path $LogDir 'launcher.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    $state = Read-RuntimeState
    $owned = Get-OwnedProcess $state
    if ($owned) {
        & taskkill.exe /PID $owned.ProcessId /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) { throw '停止进程失败。' }
        Write-Host "本地交互页已停止。"
    } else { Write-Host "没有发现本包正在运行的交互进程。" }
    Clear-RuntimeState
    exit 0
} catch {
    Write-Host "停止失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally { if ($lock) { $lock.Dispose() } }
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


STOP_BAT = r'''@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_offline.ps1" %*
set "launcher_exit=%errorlevel%"
if not "%launcher_exit%"=="0" pause
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

    write_text(staging / "README_先看这里.md", README)
    write_text(staging / "common_offline.ps1", COMMON_PS1, bom=True)
    write_text(staging / "start_offline.ps1", START_PS1, bom=True)
    write_text(staging / "stop_offline.ps1", STOP_PS1, bom=True)
    write_batch(staging / "启动离线演示.bat", START_BAT)
    write_batch(staging / "停止离线演示.bat", STOP_BAT)
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
        "entrypoints": {
            "showcase": "showcase/index.html",
            "interactive": "启动离线演示.bat",
            "stop": "停止离线演示.bat",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
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
    manifest = build(args.project_root, args.source_package, args.output, args.staging)
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
