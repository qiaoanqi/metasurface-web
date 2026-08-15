#!/usr/bin/env python3
"""Keep the paper 2 controller alive without starting scientific work.

The watchdog owns one process lock, starts one ``pipeline_supervisor --watch``
child, records only operational state, and restarts the controller after a
transient process exit. It never edits pools, papers, gates, or training files.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".state"
LOCK_PATH = STATE / "paper2_watchdog.lock"
STATUS_PATH = STATE / "paper2_watchdog_status.json"
CONTROLLER_STATE_PATH = STATE / "controller_state.json"
STDOUT_PATH = STATE / "controller_stdout.log"
STDERR_PATH = STATE / "controller_stderr.log"

sys.path.insert(0, str(ROOT))
from pipeline_supervisor import atomic_json  # noqa: E402


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def write_status(**updates) -> None:
    current = {}
    if STATUS_PATH.is_file():
        try:
            current = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
    current.update(updates)
    current["updated_at"] = now_iso()
    atomic_json(STATUS_PATH, current)


def controller_state_healthy(
    started_at: float, stale_after: float, *, now: float | None = None
) -> bool:
    """Return whether the supervised controller has refreshed its state recently."""
    current_time = time.time() if now is None else now
    try:
        state_mtime = CONTROLLER_STATE_PATH.stat().st_mtime
    except OSError:
        # A freshly started controller gets a grace period to create its state.
        return current_time - started_at < stale_after
    return current_time - max(started_at, state_mtime) < stale_after


def start_controller(interval: int) -> subprocess.Popen:
    STATE.mkdir(parents=True, exist_ok=True)
    stdout = STDOUT_PATH.open("ab")
    stderr = STDERR_PATH.open("ab")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "pipeline_supervisor.py"), "--watch", "--interval", str(interval)],
            cwd=str(ROOT),
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    finally:
        stdout.close()
        stderr.close()
    return process


def run(interval: int, restart_delay: int, max_restarts_per_hour: int) -> int:
    lock_handle = acquire_lock(LOCK_PATH)
    if lock_handle is None:
        write_status(status="already_running", lock=str(LOCK_PATH))
        return 0

    child = None
    child_started_at = 0.0
    restart_times: list[float] = []
    stopping = False
    stale_after = max(180, interval * 10)

    def stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, stop)

    try:
        write_status(status="starting", watchdog_pid=os.getpid(), interval_seconds=interval)
        while not stopping:
            now = time.time()
            restart_times[:] = [stamp for stamp in restart_times if now - stamp < 3600]
            if child is None or child.poll() is not None:
                if child is not None:
                    write_status(
                        status="controller_exited",
                        watchdog_pid=os.getpid(),
                        controller_pid=child.pid,
                        controller_returncode=child.returncode,
                        restart_count=len(restart_times),
                    )
                if len(restart_times) >= max_restarts_per_hour:
                    write_status(
                        status="backing_off",
                        watchdog_pid=os.getpid(),
                        restart_count=len(restart_times),
                        next_restart_after_seconds=restart_delay,
                    )
                    time.sleep(max(30, restart_delay))
                    continue
                if child is not None:
                    time.sleep(max(5, restart_delay))
                child = start_controller(interval)
                child_started_at = time.time()
                restart_times.append(time.time())
                write_status(
                    status="running",
                    watchdog_pid=os.getpid(),
                    controller_pid=child.pid,
                    restart_count=len(restart_times),
                )
            elif not controller_state_healthy(child_started_at, stale_after):
                write_status(
                    status="controller_stale",
                    watchdog_pid=os.getpid(),
                    controller_pid=child.pid,
                    restart_count=len(restart_times),
                    stale_after_seconds=stale_after,
                )
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)
                child = None
                child_started_at = 0.0
            else:
                write_status(
                    status="running",
                    watchdog_pid=os.getpid(),
                    controller_pid=child.pid,
                    controller_returncode=None,
                    restart_count=len(restart_times),
                )
            time.sleep(max(5, min(interval, 60)))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
        write_status(status="stopped", watchdog_pid=os.getpid())
        lock_handle.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--restart-delay", type=int, default=30)
    parser.add_argument("--max-restarts-per-hour", type=int, default=6)
    args = parser.parse_args()
    return run(
        max(5, args.interval),
        max(5, args.restart_delay),
        max(1, args.max_restarts_per_hour),
    )


if __name__ == "__main__":
    raise SystemExit(main())
