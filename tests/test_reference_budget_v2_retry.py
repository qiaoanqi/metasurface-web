import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import prepare_reference_budget_v2_retry as retry
from scripts import run_reference_resolution_budget_v2 as budget


def cases():
    return [
        {"L": 120.0 + i, "W": 100.0 + i, "H": 300.0 + i, "P": 400.0 + i}
        for i in range(8)
    ]


def valid_result(task):
    samples = len(task["wavelength_nm"])
    return {
        **task,
        "status": "ok",
        "R": np.full(samples, 0.5),
        "T": np.full(samples, 0.5),
        "time_s": 1.0,
    }


class ReferenceBudgetV2RetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old_root = retry.ROOT
        retry.ROOT = self.root
        self.addCleanup(setattr, retry, "ROOT", self.old_root)
        self.tasks = budget.build_tasks(cases())
        self.active = {"request_id": "stable-request", "attempt": 2}
        self.expected = {
            "version": budget.VERSION,
            "request": self.active,
            "plan_sha256": "P" * 64,
            "pool_sha256": "A" * 64,
            "selected_geometries": cases(),
            "expected_tasks": len(self.tasks),
            "tasks": [
                {
                    key: task[key]
                    for key in (
                        "id",
                        "geometry_index",
                        "pol",
                        "requested_nG",
                        "Nxy",
                        "step_nm",
                    )
                }
                for task in self.tasks
            ],
            "runtime_hashes": {"runner": "R" * 64},
        }
        self.checkpoint = self.root / ".state/checkpoint.pkl"
        self.checkpoint.parent.mkdir()
        payload = {
            "meta": dict(self.expected) | {
                "request": {"request_id": "stable-request", "attempt": 1}
            },
            "results": {self.tasks[0]["id"]: valid_result(self.tasks[0])},
        }
        self.checkpoint.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        self.evidence = self.root / ".state/evidence.json"
        self.ack = {
            "request_id": "stable-request",
            "attempt": 2,
            "status": "claimed",
            "worker_pid": None,
        }

    def prepare(self):
        return retry.prepare_retry(
            self.checkpoint,
            self.evidence,
            self.root / ".state",
            self.active,
            self.ack,
            self.expected,
            self.tasks,
        )

    def test_later_attempt_rebinds_valid_partial_checkpoint_once(self):
        first = self.prepare()
        second = self.prepare()
        with self.checkpoint.open("rb") as handle:
            checkpoint = pickle.load(handle)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "ready")
        self.assertEqual(checkpoint["meta"]["request"], self.active)
        self.assertEqual(len(checkpoint["results"]), 1)
        journals = list((self.root / ".state").glob("*retry_stable-request_a1_to_a2.json"))
        self.assertEqual(len(journals), 1)
        self.assertEqual(json.loads(journals[0].read_text())["status"], "completed")

    def test_missing_checkpoint_starts_fresh_but_orphan_evidence_fails(self):
        self.checkpoint.unlink()
        result = self.prepare()
        self.assertEqual(result["status"], "start_fresh")
        self.evidence.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "without its checkpoint"):
            self.prepare()

    def test_prepared_journal_is_finalized_after_checkpoint_replace(self):
        self.prepare()
        journal = next(
            (self.root / ".state").glob("*retry_stable-request_a1_to_a2.json")
        )
        payload = json.loads(journal.read_text(encoding="utf-8"))
        payload["status"] = "prepared"
        journal.write_text(json.dumps(payload), encoding="utf-8")
        result = self.prepare()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(json.loads(journal.read_text())["status"], "completed")

    def test_cross_request_and_newer_attempt_are_rejected(self):
        with self.checkpoint.open("rb") as handle:
            checkpoint = pickle.load(handle)
        for request in (
            {"request_id": "other-request", "attempt": 1},
            {"request_id": "stable-request", "attempt": 3},
        ):
            with self.subTest(request=request):
                changed = dict(checkpoint)
                changed["meta"] = dict(checkpoint["meta"]) | {"request": request}
                self.checkpoint.write_bytes(
                    pickle.dumps(changed, protocol=pickle.HIGHEST_PROTOCOL)
                )
                with self.assertRaisesRegex(ValueError, "different or newer"):
                    self.prepare()

    def test_live_worker_blocks_checkpoint_rebind(self):
        self.ack["worker_pid"] = os.getpid()
        self.ack["status"] = "running"
        with self.assertRaisesRegex(ValueError, "worker is alive"):
            self.prepare()

    def test_completed_older_evidence_is_audited_without_rebinding(self):
        before = retry.file_digest(self.checkpoint)
        self.evidence.write_text(
            json.dumps(
                {
                    "request": {"request_id": "stable-request", "attempt": 1},
                    "checkpoint": {
                        "path": ".state/checkpoint.pkl",
                        "sha256": before,
                    },
                }
            ),
            encoding="utf-8",
        )
        result = self.prepare()
        self.assertEqual(result["status"], "audit_existing_evidence")
        self.assertEqual(retry.file_digest(self.checkpoint), before)


if __name__ == "__main__":
    unittest.main()
