import copy
import json
import pickle
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

import pipeline_supervisor as supervisor


def make_policy(path: str, expected_records: int = 2):
    policy = copy.deepcopy(supervisor.load_policy())
    policy["pool"]["path"] = path
    policy["pool"]["expected_records"] = expected_records
    policy["pool"]["expected_meta"]["n_samples"] = expected_records // 2
    return policy


def make_record(pol: str, L: float = 200.0, W: float = 150.0):
    wl = np.arange(380.0, 785.0, 5.0)
    refl = np.linspace(0.1, 0.3, wl.size)
    tran = 1.0 - refl
    return {
        "L": L,
        "W": W,
        "H": 300.0,
        "P": 400.0,
        "r": max(L, W) / min(L, W),
        "pol": pol,
        "material": "TiO2",
        "substrate": "SiO2",
        "nG_actual": 131,
        "retry_nG": 131,
        "isolated": False,
        "wl_nm": wl,
        "R": refl,
        "T": tran,
        "R_plus_T_mean": 1.0,
        "quality_pass": True,
        "success": True,
    }


class PoolAuditTests(unittest.TestCase):
    def write_pool(self, root: Path, records):
        policy = make_policy("pool.pkl", len(records))
        with (root / "pool.pkl").open("wb") as handle:
            pickle.dump({"meta": copy.deepcopy(policy["pool"]["expected_meta"]), "records": records}, handle)
        return policy

    def run_audit(self, root, policy):
        old_root = supervisor.ROOT
        supervisor.ROOT = root
        try:
            return supervisor.audit_pool(root / "pool.pkl", policy["pool"])
        finally:
            supervisor.ROOT = old_root

    def test_valid_pair_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.write_pool(root, [make_record("p"), make_record("s")])
            result = self.run_audit(root, policy)
            self.assertTrue(result["passed"])
            self.assertEqual(result["complete_pairs"], 1)

    def test_nan_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = make_record("p")
            broken["R"][0] = np.nan
            policy = self.write_pool(root, [broken, make_record("s")])
            result = self.run_audit(root, policy)
            self.assertFalse(result["passed"])
            self.assertIn("NONFINITE_SPECTRUM", {item["code"] for item in result["errors"]})

    def test_duplicate_and_missing_pair_are_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.write_pool(root, [make_record("p"), make_record("p")])
            result = self.run_audit(root, policy)
            codes = {item["code"] for item in result["errors"]}
            self.assertFalse(result["passed"])
            self.assertIn("DUPLICATE_KEYS", codes)
            self.assertIn("INCOMPLETE_POLARIZATION_PAIRS", codes)

    def test_pointwise_error_cannot_hide_in_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = make_record("p")
            broken["T"][0] -= 0.1
            broken["T"][1] += 0.1
            policy = self.write_pool(root, [broken, make_record("s")])
            result = self.run_audit(root, policy)
            self.assertFalse(result["passed"])
            self.assertIn("POINTWISE_CONSERVATION_FAIL", {item["code"] for item in result["errors"]})

    def test_incomplete_checkpoint_can_be_healthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.write_pool(root, [make_record("p")])
            policy["pool"]["expected_records"] = 2
            result = self.run_audit(root, policy)
            self.assertFalse(result["passed"])
            self.assertTrue(result["healthy_checkpoint"])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_globals = {
            name: getattr(supervisor, name)
            for name in (
                "ROOT", "STATE", "STATUS", "POLICY", "AUDIT_RESULT", "NEXT_PLAN",
                "CONTROLLER_STATE", "DISPATCH_REQUEST", "EXECUTOR_ACK", "LEGACY_INBOX",
                "GATE_STATE",
            )
        }
        supervisor.ROOT = self.root
        supervisor.STATE = self.root / ".state"
        supervisor.STATUS = supervisor.STATE / "hermes_status.json"
        supervisor.POLICY = self.root / "pipeline_policy.json"
        supervisor.AUDIT_RESULT = supervisor.STATE / "audit_result.json"
        supervisor.NEXT_PLAN = supervisor.STATE / "next_plan.json"
        supervisor.CONTROLLER_STATE = supervisor.STATE / "controller_state.json"
        supervisor.DISPATCH_REQUEST = supervisor.STATE / "dispatch_request.json"
        supervisor.EXECUTOR_ACK = supervisor.STATE / "executor_ack.json"
        supervisor.LEGACY_INBOX = supervisor.STATE / "hermes_inbox.json"
        supervisor.GATE_STATE = supervisor.STATE / "gate_state.json"
        supervisor.STATE.mkdir()

        self.policy = make_policy("pool.pkl", 2)
        self.policy["protected_files"] = []
        with (self.root / "pool.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "meta": copy.deepcopy(self.policy["pool"]["expected_meta"]),
                    "records": [make_record("p"), make_record("s")],
                },
                handle,
            )

    def tearDown(self):
        for name, value in self.old_globals.items():
            setattr(supervisor, name, value)
        self.tmp.cleanup()

    def write_status(self, status):
        supervisor.atomic_json(supervisor.STATUS, status)

    def test_stale_running_is_reconciled_from_artifact(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            result = supervisor.evaluate_once(self.policy)
        self.assertEqual(result["effective_status"], "completed")
        self.assertTrue(result["status_reconciled"])
        self.assertEqual(result["next_action"], "pool_validation")
        self.assertFalse(result["training_allowed"])

    def test_failed_producer_is_never_relabelled_passed(self):
        self.write_status({"status": "failed", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            result = supervisor.evaluate_once(self.policy)
        audit = supervisor.load_json(supervisor.AUDIT_RESULT)
        self.assertEqual(result["controller_status"], "blocked")
        self.assertEqual(result["next_action"], "stop_and_report")
        self.assertFalse(audit["passed"])

    def test_ack_without_gate_evidence_retries_same_request(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": request["request_id"],
                "attempt": request["attempt"],
                "status": "completed",
                "observed_at": supervisor.now_iso(),
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["request_id"], request["request_id"])
        self.assertEqual(second["dispatch"]["status"], "pending")
        self.assertEqual(second["dispatch"]["attempt"], 2)

    def test_matching_ack_and_evidence_advance_to_next_gate(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        audit = supervisor.load_json(supervisor.AUDIT_RESULT)
        manifest = self.root / "pool_manifest.json"
        supervisor.atomic_json(
            manifest,
            {
                "pool_sha256": audit["pool"]["sha256"],
                "records": audit["pool"]["records"],
                "strict_validation_passed": True,
                "immutable": True,
            },
        )
        supervisor.atomic_json(
            supervisor.GATE_STATE,
            {
                "schema_version": 1,
                "gates": {
                    "pool_manifest_frozen": {
                        "passed": True,
                        "checked_at": supervisor.now_iso(),
                        "evidence": [
                            {"path": "pool_manifest.json", "sha256": supervisor.file_digest(manifest)}
                        ],
                    }
                },
            },
        )
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": request["request_id"],
                "attempt": request["attempt"],
                "status": "completed",
                "observed_at": supervisor.now_iso(),
                "outputs": [{"path": ".state/pool_manifest.json", "material": "pool_manifest"}],
                "paper_hashes": [
                    {
                        "path": ".state/pool_manifest.json",
                        "md5": supervisor.file_digest(manifest, "md5"),
                    }
                ],
                "checks": {"pool_sha256": audit["pool"]["sha256"]},
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertNotEqual(second["dispatch"]["request_id"], request["request_id"])
        self.assertEqual(second["dispatch"]["action"], "d65_colorimetry")
        self.assertEqual(second["dispatch"]["attempt"], 1)

    def test_completed_ack_with_wrong_paper_hash_does_not_advance(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        audit = supervisor.load_json(supervisor.AUDIT_RESULT)
        output = self.root / "output.json"
        supervisor.atomic_json(output, {"passed": True})
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": request["request_id"],
                "attempt": request["attempt"],
                "status": "completed",
                "observed_at": supervisor.now_iso(),
                "outputs": [{"path": "output.json", "material": "evidence"}],
                "paper_hashes": [{"path": "output.json", "md5": "0" * 32}],
                "checks": {"pool_sha256": audit["pool"]["sha256"]},
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["status"], "pending")
        self.assertEqual(second["dispatch"]["attempt"], 2)
        self.assertIn("paper hash mismatch", second["dispatch"]["last_error"])

    def test_timeout_retries_then_stops(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            result = supervisor.evaluate_once(self.policy)
        request = result["dispatch"]
        old = (datetime.now().astimezone() - timedelta(hours=1)).isoformat(timespec="seconds")
        for expected_attempt in (2, 3):
            request["updated_at"] = old
            supervisor.atomic_json(supervisor.DISPATCH_REQUEST, request)
            with patch.object(supervisor, "pid_alive", return_value=False):
                result = supervisor.evaluate_once(self.policy)
            request = result["dispatch"]
            self.assertEqual(request["attempt"], expected_attempt)
            self.assertEqual(request["status"], "pending")
        request["updated_at"] = old
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, request)
        with patch.object(supervisor, "pid_alive", return_value=False):
            result = supervisor.evaluate_once(self.policy)
        self.assertEqual(result["dispatch"]["status"], "failed")
        self.assertEqual(result["next_action"], "stop_and_report")

    def test_gate_requires_matching_evidence_hash(self):
        evidence = self.root / "evidence.json"
        evidence.write_text(
            json.dumps(
                {
                    "passed": True,
                    "pool_sha256": "ABC",
                    "evidence_version": "paper2-d65-v1",
                }
            ) + "\n",
            encoding="ascii",
        )
        supervisor.atomic_json(
            supervisor.GATE_STATE,
            {
                "schema_version": 1,
                "gates": {
                    "d65_colorimetry": {
                        "passed": True,
                        "checked_at": supervisor.now_iso(),
                        "evidence": [
                            {"path": "evidence.json", "sha256": supervisor.file_digest(evidence)}
                        ],
                    }
                },
            },
        )
        gates, _ = supervisor.verify_gate_evidence(
            self.policy, {"passed": True, "sha256": "ABC", "records": 2}
        )
        self.assertTrue(gates["d65_colorimetry"])
        evidence.write_text(
            json.dumps(
                {
                    "passed": False,
                    "pool_sha256": "ABC",
                    "evidence_version": "paper2-d65-v1",
                }
            ) + "\n",
            encoding="ascii",
        )
        gates, _ = supervisor.verify_gate_evidence(
            self.policy, {"passed": True, "sha256": "ABC", "records": 2}
        )
        self.assertFalse(gates["d65_colorimetry"])

    def test_gate_rejects_wrong_evidence_version(self):
        evidence = self.root / "evidence.json"
        evidence.write_text(
            json.dumps(
                {
                    "passed": True,
                    "pool_sha256": "ABC",
                    "evidence_version": "paper2-d65-v0",
                }
            ) + "\n",
            encoding="ascii",
        )
        supervisor.atomic_json(
            supervisor.GATE_STATE,
            {
                "schema_version": 1,
                "gates": {
                    "d65_colorimetry": {
                        "passed": True,
                        "checked_at": supervisor.now_iso(),
                        "evidence": [
                            {"path": "evidence.json", "sha256": supervisor.file_digest(evidence)}
                        ],
                    }
                },
            },
        )
        gates, details = supervisor.verify_gate_evidence(
            self.policy, {"passed": True, "sha256": "ABC", "records": 2}
        )
        self.assertFalse(gates["d65_colorimetry"])
        self.assertIn("evidence_version", details["d65_colorimetry"]["evidence"][0]["semantic_error"])

    def test_valid_pool_manifest_is_derived_without_manual_gate_entry(self):
        manifest = supervisor.STATE / "pool_manifest.json"
        supervisor.atomic_json(
            manifest,
            {
                "pool_sha256": "ABC",
                "records": 2,
                "strict_validation_passed": True,
                "immutable": True,
            },
        )
        gates, details = supervisor.verify_gate_evidence(
            self.policy, {"passed": True, "sha256": "ABC", "records": 2}
        )
        self.assertTrue(gates["pool_manifest_frozen"])
        self.assertTrue(details["pool_manifest_frozen"]["verified"])

    def test_training_action_is_unreachable_until_all_prerequisites_pass(self):
        gates = {item["gate"]: False for item in self.policy["workflow"]["actions"]}
        gates.update({"pool_complete": True, "strict_pool_validation": True, "training_allowed": False})
        self.assertEqual(supervisor.select_workflow_action(self.policy, gates), "pool_validation")
        for gate in self.policy["workflow"]["required_before_training"]:
            gates[gate] = True
        gates["training_allowed"] = True
        self.assertEqual(supervisor.select_workflow_action(self.policy, gates), "training_pilot")

    def test_running_ack_with_live_lease_prevents_retry(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        lease = (datetime.now().astimezone() + timedelta(hours=1)).isoformat(timespec="seconds")
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": request["request_id"],
                "attempt": request["attempt"],
                "status": "running",
                "observed_at": supervisor.now_iso(),
                "lease_expires_at": lease,
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["status"], "in_progress")
        self.assertEqual(second["dispatch"]["attempt"], 1)

    def test_expired_running_lease_retries(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        lease = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat(timespec="seconds")
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": request["request_id"],
                "attempt": request["attempt"],
                "status": "running",
                "observed_at": supervisor.now_iso(),
                "lease_expires_at": lease,
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["status"], "pending")
        self.assertEqual(second["dispatch"]["attempt"], 2)

    def test_dead_worker_does_not_hide_behind_future_lease(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        future = (datetime.now().astimezone() + timedelta(hours=1)).isoformat(timespec="seconds")
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": request["request_id"],
                "attempt": request["attempt"],
                "status": "running",
                "observed_at": supervisor.now_iso(),
                "lease_expires_at": future,
                "worker_pid": 999999,
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["status"], "pending")
        self.assertEqual(second["dispatch"]["attempt"], 2)

    def test_scientific_failure_is_terminal_not_retried(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": request["request_id"],
                "attempt": request["attempt"],
                "status": "failed",
                "failure_class": "scientific",
                "error": "pre-registered convergence threshold failed",
                "observed_at": supervisor.now_iso(),
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["status"], "failed")
        self.assertEqual(second["dispatch"]["attempt"], 1)
        self.assertEqual(second["next_action"], "stop_and_report")

    def test_scientific_failure_generates_guarded_recovery_plan(self):
        plan = supervisor.build_recovery_plan(
            "joint_numerical_convergence",
            {
                "request_id": "failed-request",
                "status": "failed",
                "failure_class": "scientific",
                "terminal_failure": True,
                "last_error": "cross-order mismatch",
            },
            self.policy,
        )
        self.assertEqual(plan["status"], "terminal_review")
        self.assertFalse(plan["automatic_retry"])
        self.assertEqual(
            plan["recommended_strategy"],
            "inspect_failed_geometry_then_rerun_frozen_case",
        )
        self.assertIn("do not change pre-registered thresholds", plan["guardrails"])

    def test_running_dispatch_generates_monitoring_plan(self):
        plan = supervisor.build_recovery_plan(
            "cross_solver_spectrum_validation",
            {"request_id": "running-request", "status": "in_progress"},
            self.policy,
        )
        self.assertEqual(plan["status"], "monitoring")
        self.assertFalse(plan["automatic_retry"])

    def test_evidence_backed_strategy_can_retry_same_failed_gate(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        failed = first["dispatch"]
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": failed["request_id"],
                "attempt": failed["attempt"],
                "status": "failed",
                "failure_class": "scientific",
                "error": "convergence gate failed",
                "observed_at": supervisor.now_iso(),
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            terminal = supervisor.evaluate_once(self.policy)
        self.assertEqual(terminal["dispatch"]["status"], "failed")

        repair = self.root / "repair_evidence.json"
        repair.write_text('{"passed": true}\n', encoding="ascii")
        self.policy["strategy_override"] = {
            "enabled": True,
            "decision": "retry_same_gate",
            "revision": 1,
            "action": failed["action"],
            "based_on_request_id": failed["request_id"],
            "instruction_append": "Retry with the validated implementation repair; do not change thresholds.",
            "evidence": [
                {"path": "repair_evidence.json", "sha256": supervisor.file_digest(repair)}
            ],
        }
        with patch.object(supervisor, "pid_alive", return_value=False):
            retried = supervisor.evaluate_once(self.policy)
        self.assertNotEqual(retried["dispatch"]["request_id"], failed["request_id"])
        self.assertEqual(retried["dispatch"]["status"], "pending")
        self.assertEqual(retried["dispatch"]["attempt"], 1)
        self.assertEqual(retried["dispatch"]["strategy_revision"], 1)


if __name__ == "__main__":
    unittest.main()
