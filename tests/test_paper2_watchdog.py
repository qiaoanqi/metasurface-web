import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import pipeline_supervisor as supervisor


def load_watchdog():
    path = Path(__file__).parents[1] / "scripts" / "paper2_watchdog.py"
    spec = importlib.util.spec_from_file_location("paper2_watchdog_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class WatchdogTests(unittest.TestCase):
    def test_atomic_json_leaves_no_shared_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            supervisor.atomic_json(path, {"passed": True, "n": 1})
            supervisor.atomic_json(path, {"passed": True, "n": 2})
            self.assertEqual(json.loads(path.read_text(encoding="ascii"))["n"], 2)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_watchdog_status_is_atomic_and_preserves_fields(self):
        watchdog = load_watchdog()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watchdog.STATUS_PATH = root / "status.json"
            watchdog.LOCK_PATH = root / "watchdog.lock"
            watchdog.write_status(status="running", controller_pid=123)
            watchdog.write_status(status="running", controller_pid=456)
            payload = json.loads(watchdog.STATUS_PATH.read_text(encoding="ascii"))
            self.assertEqual(payload["status"], "running")
            self.assertEqual(payload["controller_pid"], 456)
            self.assertIn("updated_at", payload)

    def test_watchdog_lock_is_exclusive_and_releases(self):
        watchdog = load_watchdog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watchdog.lock"
            first = watchdog.acquire_lock(path)
            self.assertIsNotNone(first)
            first.close()
            second = watchdog.acquire_lock(path)
            self.assertIsNotNone(second)
            second.close()

    def test_controller_state_health_has_startup_grace(self):
        watchdog = load_watchdog()
        with tempfile.TemporaryDirectory() as tmp:
            watchdog.CONTROLLER_STATE_PATH = Path(tmp) / "controller_state.json"
            self.assertTrue(
                watchdog.controller_state_healthy(100.0, 180.0, now=200.0)
            )
            self.assertFalse(
                watchdog.controller_state_healthy(100.0, 180.0, now=281.0)
            )

    def test_controller_state_health_uses_latest_refresh(self):
        watchdog = load_watchdog()
        with tempfile.TemporaryDirectory() as tmp:
            watchdog.CONTROLLER_STATE_PATH = Path(tmp) / "controller_state.json"
            watchdog.CONTROLLER_STATE_PATH.write_text("{}", encoding="ascii")
            os.utime(watchdog.CONTROLLER_STATE_PATH, (250.0, 250.0))
            self.assertTrue(
                watchdog.controller_state_healthy(100.0, 180.0, now=400.0)
            )
            self.assertFalse(
                watchdog.controller_state_healthy(100.0, 180.0, now=431.0)
            )


if __name__ == "__main__":
    unittest.main()
