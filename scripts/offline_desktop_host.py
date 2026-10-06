"""Own the offline app window and service; closing the window stops both.

Only the Python standard library and an installed Edge/Chrome are needed.
Child processes enter Windows jobs before their first instruction, so even an
unexpected host exit cannot leave a detached service holding package files.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes as w
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", w.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD),
                ("SchedulingClass", w.DWORD)]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class ACCOUNTING(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in (
        "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")]
    _fields_ += [(name, w.DWORD) for name in (
        "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]


class STARTUPINFO(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("lpReserved", w.LPWSTR), ("lpDesktop", w.LPWSTR),
                ("lpTitle", w.LPWSTR), ("dwX", w.DWORD), ("dwY", w.DWORD),
                ("dwXSize", w.DWORD), ("dwYSize", w.DWORD), ("dwXCountChars", w.DWORD),
                ("dwYCountChars", w.DWORD), ("dwFillAttribute", w.DWORD),
                ("dwFlags", w.DWORD), ("wShowWindow", w.WORD), ("cbReserved2", w.WORD),
                ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
                ("hStdInput", w.HANDLE), ("hStdOutput", w.HANDLE), ("hStdError", w.HANDLE)]


class PROCESSINFO(ctypes.Structure):
    _fields_ = [("hProcess", w.HANDLE), ("hThread", w.HANDLE),
                ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD)]


k32 = ctypes.WinDLL("kernel32", use_last_error=True)


def signature(name, args, result):
    fn = getattr(k32, name)
    fn.argtypes = args
    fn.restype = result
    return fn


CreateJob = signature("CreateJobObjectW", [ctypes.c_void_p, w.LPCWSTR], w.HANDLE)
SetJob = signature("SetInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL)
QueryJob = signature("QueryInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL)
AssignJob = signature("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL)
CloseHandle = signature("CloseHandle", [w.HANDLE], w.BOOL)
CreateProcess = signature("CreateProcessW", [w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
    w.BOOL, w.DWORD, ctypes.c_void_p, w.LPCWSTR, ctypes.POINTER(STARTUPINFO), ctypes.POINTER(PROCESSINFO)], w.BOOL)
ResumeThread = signature("ResumeThread", [w.HANDLE], w.DWORD)
TerminateProcess = signature("TerminateProcess", [w.HANDLE, w.UINT], w.BOOL)
WaitProcess = signature("WaitForSingleObject", [w.HANDLE, w.DWORD], w.DWORD)
GetCurrentProcess = signature("GetCurrentProcess", [], w.HANDLE)
GetProcessTimes = signature("GetProcessTimes", [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4, w.BOOL)
GetModuleFileName = signature("GetModuleFileNameW", [w.HANDLE, w.LPWSTR, w.DWORD], w.DWORD)


def checked(value):
    if not value:
        raise ctypes.WinError(ctypes.get_last_error())
    return value


class OwnedJob:
    def __init__(self):
        self.handle = checked(CreateJob(None, None))
        self.process_handles = []
        limits = EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        try:
            checked(SetJob(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def spawn(self, argv, cwd, stdout=None, stderr=None):
        import msvcrt
        startup = STARTUPINFO()
        startup.cb = ctypes.sizeof(startup)
        inherited = []
        null_in = None
        if stdout is not None:
            null_in = open(os.devnull, "rb")
            startup.dwFlags = 0x100  # STARTF_USESTDHANDLES
            handles = [msvcrt.get_osfhandle(f.fileno()) for f in (null_in, stdout, stderr)]
            for handle in handles:
                os.set_handle_inheritable(handle, True)
                inherited.append(handle)
            startup.hStdInput, startup.hStdOutput, startup.hStdError = handles
        info = PROCESSINFO()
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline([str(a) for a in argv]))
        flags = 0x4 | (0x08000000 if stdout is not None else 0)  # suspended + no console
        try:
            checked(CreateProcess(str(argv[0]), command, None, None, bool(inherited), flags,
                None, str(cwd), ctypes.byref(startup), ctypes.byref(info)))
            try:
                checked(AssignJob(self.handle, info.hProcess))
                if ResumeThread(info.hThread) == 0xFFFFFFFF:
                    raise ctypes.WinError(ctypes.get_last_error())
                self.process_handles.append(info.hProcess)
                return int(info.dwProcessId)
            except BaseException:
                TerminateProcess(info.hProcess, 1)
                CloseHandle(info.hProcess)
                raise
            finally:
                CloseHandle(info.hThread)
        finally:
            for handle in inherited:
                os.set_handle_inheritable(handle, False)
            if null_in:
                null_in.close()

    def active(self):
        info = ACCOUNTING()
        checked(QueryJob(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return int(info.ActiveProcesses)

    def close(self):
        if self.handle:
            CloseHandle(self.handle)
            self.handle = None
        for handle in self.process_handles:
            WaitProcess(handle, 5000)
            CloseHandle(handle)
        self.process_handles.clear()


def creation_time():
    times = [w.FILETIME() for _ in range(4)]
    checked(GetProcessTimes(GetCurrentProcess(), *(ctypes.byref(t) for t in times)))
    ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    value = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond:06d}0Z"


def actual_executable():
    buffer = ctypes.create_unicode_buffer(32768)
    checked(GetModuleFileName(None, buffer, len(buffer)))
    return buffer.value


def browser_executable():
    locations = []
    for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES"), os.environ.get("LOCALAPPDATA")):
        if base:
            locations.extend([Path(base) / "Microsoft/Edge/Application/msedge.exe",
                              Path(base) / "Google/Chrome/Application/chrome.exe"])
    for path in locations:
        if path.is_file():
            return path
    raise RuntimeError("需要已安装的 Microsoft Edge 或 Google Chrome。")


def free_port(requested):
    for port in range(requested, min(requested + 21, 65536)):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    raise RuntimeError("本地端口已被占用。")


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run(port, no_browser):
    import msvcrt
    root = Path(__file__).resolve().parent
    runtime = root / "runtime"
    logs = root / "logs"
    state_path = logs / "runtime-state.json"
    server_job, browser_job = None, None
    state = None
    profile = None
    logs.mkdir(exist_ok=True)
    logging.basicConfig(filename=logs / "desktop-host.log", encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    with (logs / "desktop-host.lock").open("a+b") as host_lock:
        host_lock.seek(0)
        host_lock.write(b"0")
        host_lock.flush()
        host_lock.seek(0)
        msvcrt.locking(host_lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            browser = None if no_browser else browser_executable()
            port = free_port(port)
            server_job = OwnedJob()
            with (logs / "streamlit.out.log").open("wb") as out, (logs / "streamlit.err.log").open("wb") as err:
                python = root / ".venv/Scripts/python.exe"
                server_pid = server_job.spawn([python, "-m", "streamlit", "run", "app.py",
                    "--server.address", "127.0.0.1", "--server.port", str(port),
                    "--server.headless", "true", "--browser.gatherUsageStats", "false"], runtime, out, err)
                state = dict(mode="desktop_window", root=str(root), runtime=str(runtime),
                    executable=str(python), host_executable=actual_executable(), process_id=os.getpid(),
                    server_process_id=server_pid, created_at_utc=creation_time(), port=port, status="starting")
                write_state(state_path, state)
                url = f"http://127.0.0.1:{port}/"
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if server_job.active() == 0:
                        raise RuntimeError("交互服务提前退出，请查看 streamlit.err.log。")
                    try:
                        with urlopen(url + "_stcore/health", timeout=1) as response:
                            if response.status == 200 and response.read().strip() == b"ok":
                                break
                    except OSError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError("交互服务未能在 90 秒内就绪。")
                (root / "showcase/local-demo-config.js").write_text(f"window.OFFLINE_APP_URL = '{url}';\n", encoding="ascii")
                if browser:
                    # Unique profile prevents attaching to the user's normal browser.
                    profile = Path(tempfile.mkdtemp(prefix="metasurface-app-"))
                    state["browser_profile"] = str(profile)
                    browser_job = OwnedJob()
                    state["browser_process_id"] = browser_job.spawn([browser, f"--app={url}",
                        f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
                        "--disable-background-mode", "--disable-background-networking",
                        "--disable-extensions", "--disable-features=msStartupBoost", "--window-size=1360,900"], root)
                    time.sleep(0.5)
                    if browser_job.active() == 0:
                        raise RuntimeError("应用窗口提前退出，请查看 desktop-host.log。")
                state["status"] = "ready"
                write_state(state_path, state)
                logging.info("ready port=%s host=%s server=%s", port, os.getpid(), server_pid)
                while not browser_job or browser_job.active() > 0:
                    if server_job.active() == 0:
                        raise RuntimeError("交互服务已停止。")
                    time.sleep(0.2)
                logging.info("window closed; stopping service")
        except BaseException:
            logging.exception("offline app stopped with an error")
            raise
        finally:
            if browser_job:
                browser_job.close()
            if server_job:
                server_job.close()
            if profile is not None:
                # Only our fresh, unique profile under the system temp directory.
                if profile.resolve().parent != Path(tempfile.gettempdir()).resolve() or not profile.name.startswith('metasurface-app-'):
                    raise RuntimeError('Unexpected temporary browser profile path')
                for attempt in range(20):
                    try:
                        shutil.rmtree(profile)
                        break
                    except OSError:
                        time.sleep(0.1)
                else:
                    logging.warning('Temporary profile remains: %s', profile)
            if state is not None:
                try:
                    current = json.loads(state_path.read_text(encoding="utf-8-sig"))
                    if current.get("process_id") == os.getpid():
                        state_path.unlink()
                        (logs / "streamlit.pid").unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
            logging.info("all owned processes stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8512)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    run(args.port, args.no_browser)
