import copy
import hashlib
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
        self.policy["immutable_assets"] = []
        self.policy["integrity"]["enforce"] = False
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

    def test_completed_ack_rejects_immutable_output_path(self):
        legacy = self.root / "legacy.pkl"
        legacy.write_bytes(b"immutable")
        evidence = self.root / "evidence.json"
        evidence.write_text("{}\n", encoding="ascii")
        self.policy["immutable_assets"] = [
            {"path": "legacy.pkl", "md5": supervisor.file_digest(legacy, "md5")}
        ]
        ack = {
            "checks": {"pool_sha256": "ABC"},
            "outputs": [{"path": "legacy.pkl", "material": "legacy"}],
            "paper_hashes": [{"path": "evidence.json", "md5": supervisor.file_digest(evidence, "md5")}],
        }
        valid, error = supervisor.validate_completed_ack(ack, "ABC", self.policy)
        self.assertFalse(valid)
        self.assertIn("immutable asset", error)

    def test_policy_integrity_lock_is_required_when_enabled(self):
        self.policy["integrity"]["enforce"] = True
        result = supervisor.verify_policy_integrity(self.policy)
        self.assertFalse(result["passed"])
        self.assertIn("lock", result["error"])

    def write_valid_active_pool_chain(self, pool_spec_updates=None):
        replacement_dir = self.root / "data" / "replacement"
        replacement_dir.mkdir(parents=True, exist_ok=True)
        replacement = replacement_dir / "replacement_v1.pkl"
        replacement.write_bytes((self.root / "pool.pkl").read_bytes())
        replacement_sha = supervisor.file_digest(replacement)

        pool_spec = copy.deepcopy(self.policy["pool"])
        pool_spec.update(
            {
                "path": "data/replacement/replacement_v1.pkl",
                "resume_command": "resume approved replacement",
            }
        )
        if pool_spec_updates:
            pool_spec.update(pool_spec_updates)

        reference = supervisor.STATE / "reference_holdout_audit.json"
        supervisor.atomic_json(
            reference,
            {
                "schema_version": 1,
                "evidence_version": "paper2-reference-holdout-audit-v1",
                "passed": True,
                "production_reference_approved": True,
                "pool_sha256": supervisor.file_digest(self.root / "pool.pkl"),
            },
        )
        reference_binding = {
            "path": ".state/reference_holdout_audit.json",
            "sha256": supervisor.file_digest(reference),
        }
        supervisor.atomic_json(
            supervisor.GATE_STATE,
            {
                "schema_version": 1,
                "gates": {
                    "reference_resolution": {
                        "passed": True,
                        "checked_at": supervisor.now_iso(),
                        "evidence": [reference_binding],
                    }
                },
            },
        )

        runtime = self.root / "replacement_runtime.py"
        runtime.write_text("# frozen replacement runtime\n", encoding="ascii")
        protocol = supervisor.STATE / "replacement_protocol.json"
        supervisor.atomic_json(
            protocol,
            {
                "schema_version": 1,
                "evidence_version": "paper2-replacement-protocol-v1",
                "approved": True,
                "automatic_launch_authorized": True,
                "source_reference_gate": reference_binding,
                "runtime_hashes": {
                    "replacement_runtime.py": supervisor.file_digest(runtime),
                },
                "pool_spec": pool_spec,
            },
        )
        approved_protocol = {
            "path": ".state/replacement_protocol.json",
            "sha256": supervisor.file_digest(protocol),
        }
        activation_id = hashlib.sha256(
            f"{approved_protocol['sha256']}|{replacement_sha}".encode("ascii")
        ).hexdigest()[:24]

        evidence = supervisor.STATE / "replacement_activation_evidence.json"
        supervisor.atomic_json(
            evidence,
            {
                "schema_version": 1,
                "evidence_version": "paper2-replacement-pool-v1",
                "passed": True,
                "pool_sha256": replacement_sha,
                "pool_spec": pool_spec,
                "approved_protocol": approved_protocol,
                "activation_id": activation_id,
            },
        )
        evidence_binding = {
            "path": ".state/replacement_activation_evidence.json",
            "sha256": supervisor.file_digest(evidence),
        }
        previous_pool = {
            "path": "pool.pkl",
            "sha256": supervisor.file_digest(self.root / "pool.pkl"),
            "md5": supervisor.file_digest(self.root / "pool.pkl", "md5"),
        }

        strict_manifest = supervisor.STATE / "replacement_pool_manifest.json"
        supervisor.atomic_json(
            strict_manifest,
            {
                "schema_version": 1,
                "immutable": True,
                "strict_validation_passed": True,
                "pool_sha256": replacement_sha,
                "records": pool_spec["expected_records"],
                "approved_protocol": approved_protocol,
                "activation_id": activation_id,
                "pool_spec_sha256": supervisor.json_payload_digest(pool_spec),
            },
        )
        active_manifest = {
            "schema_version": 1,
            "active": True,
            "pool_spec": pool_spec,
            "pool_sha256": replacement_sha,
            "activation_id": activation_id,
            "approved_protocol": approved_protocol,
            "previous_pool": previous_pool,
            "activation_evidence": evidence_binding,
            "pool_manifest": {
                "path": ".state/replacement_pool_manifest.json",
                "sha256": supervisor.file_digest(strict_manifest),
            },
        }
        supervisor.atomic_json(supervisor.STATE / "active_pool.json", active_manifest)
        return {
            "replacement": replacement,
            "reference": reference,
            "protocol": protocol,
            "evidence": evidence,
            "strict_manifest": strict_manifest,
            "active_manifest": active_manifest,
            "pool_spec": pool_spec,
        }

    def test_hash_backed_active_pool_manifest_switches_pool(self):
        chain = self.write_valid_active_pool_chain()
        spec, state = supervisor.resolve_active_pool(self.policy)
        self.assertTrue(state["passed"])
        self.assertTrue(state["active"])
        self.assertEqual(spec, chain["pool_spec"])
        chain["replacement"].write_bytes(b"tampered")
        _, state = supervisor.resolve_active_pool(self.policy)
        self.assertFalse(state["passed"])
        self.assertIn("SHA256", state["error"])

    def test_active_pool_rejects_zero_expected_records(self):
        self.write_valid_active_pool_chain({"expected_records": 0})
        _, state = supervisor.resolve_active_pool(self.policy)
        self.assertFalse(state["passed"])
        self.assertIn("expected_records", state["error"])

    def test_active_pool_rejects_any_tampered_activation_layer(self):
        cases = {
            "pool_spec": lambda chain: chain["active_manifest"]["pool_spec"].update(
                {"quality_tolerance": 0.5}
            ),
            "evidence": lambda chain: chain["evidence"].write_text(
                '{"tampered": true}\n', encoding="ascii"
            ),
            "protocol": lambda chain: chain["protocol"].write_text(
                '{"tampered": true}\n', encoding="ascii"
            ),
            "reference": lambda chain: chain["reference"].write_text(
                '{"tampered": true}\n', encoding="ascii"
            ),
            "strict_manifest": lambda chain: chain["strict_manifest"].write_text(
                '{"tampered": true}\n', encoding="ascii"
            ),
        }
        expected_errors = {
            "pool_spec": "pool_spec differs",
            "evidence": "activation evidence SHA256 mismatch",
            "protocol": "protocol SHA256 mismatch",
            "reference": "reference gate hash mismatch",
            "strict_manifest": "pool manifest hash mismatch",
        }
        for name, tamper in cases.items():
            with self.subTest(layer=name):
                chain = self.write_valid_active_pool_chain()
                tamper(chain)
                if name == "pool_spec":
                    supervisor.atomic_json(
                        supervisor.STATE / "active_pool.json", chain["active_manifest"]
                    )
                _, state = supervisor.resolve_active_pool(self.policy)
                self.assertFalse(state["passed"])
                self.assertIn(expected_errors[name], state["error"])

    def test_integrity_mismatch_preserves_active_dispatch(self):
        self.policy["integrity"]["enforce"] = True
        active = {
            "request_id": "active-request",
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "attempt": 1,
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, active)
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            result = supervisor.evaluate_once(self.policy)
        preserved = supervisor.load_json(supervisor.DISPATCH_REQUEST)
        self.assertEqual(preserved, active)
        self.assertEqual(result["dispatch"]["request_id"], "active-request")
        self.assertEqual(result["next_action"], "stop_and_report")

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
                "outputs": [{"path": "pool_manifest.json", "material": "pool_manifest"}],
                "paper_hashes": [
                    {
                        "path": "pool_manifest.json",
                        "md5": supervisor.file_digest(manifest, "md5"),
                    }
                ],
                "checks": {"pool_sha256": audit["pool"]["sha256"]},
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["request_id"], request["request_id"])
        self.assertEqual(second["dispatch"]["action"], "pool_validation")
        self.assertEqual(second["dispatch"]["status"], "acknowledged")

        with patch.object(supervisor, "pid_alive", return_value=False):
            third = supervisor.evaluate_once(self.policy)
        self.assertNotEqual(third["dispatch"]["request_id"], request["request_id"])
        self.assertEqual(third["dispatch"]["action"], "d65_colorimetry")
        history = (
            supervisor.STATE / "dispatch_history"
            / f"{request['request_id']}-attempt{request['attempt']}.json"
        )
        archived = supervisor.load_json(history)
        self.assertEqual(archived["request"]["request_id"], request["request_id"])
        self.assertEqual(archived["final_ack"]["status"], "completed")
        self.assertEqual(archived["next_action"], "d65_colorimetry")
        self.assertEqual(third["dispatch"]["attempt"], 1)

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

    def test_ack_from_wrong_executor_thread_is_terminal_policy_failure(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": request["request_id"],
                "attempt": request["attempt"],
                "thread_id": "not-the-authorized-executor",
                "status": "running",
                "observed_at": supervisor.now_iso(),
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["status"], "failed")
        self.assertTrue(second["dispatch"]["terminal_failure"])
        self.assertEqual(second["dispatch"]["failure_class"], "policy")
        self.assertIn("identity mismatch", second["dispatch"]["last_error"])

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

    def test_protocol_bound_gate_survives_active_pool_switch(self):
        evidence = self.root / "evidence.json"
        evidence.write_text(json.dumps({
            "passed": True,
            "pool_sha256": "SOURCE-POOL",
            "evidence_version": "paper2-d65-v1",
        }) + "\n", encoding="ascii")
        supervisor.atomic_json(supervisor.GATE_STATE, {
            "schema_version": 1,
            "gates": {"d65_colorimetry": {
                "passed": True,
                "checked_at": supervisor.now_iso(),
                "evidence": [{"path": "evidence.json", "sha256": supervisor.file_digest(evidence)}],
            }},
        })
        gates, _ = supervisor.verify_gate_evidence(
            self.policy, {"passed": True, "sha256": "ACTIVE-POOL", "records": 2}
        )
        self.assertTrue(gates["d65_colorimetry"])
        for item in self.policy["workflow"]["actions"]:
            if item["gate"] == "d65_colorimetry":
                item["binding"] = "pool"
        gates, _ = supervisor.verify_gate_evidence(
            self.policy, {"passed": True, "sha256": "ACTIVE-POOL", "records": 2}
        )
        self.assertFalse(gates["d65_colorimetry"])

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

    def test_workflow_actions_follow_preregistered_order(self):
        actions = self.policy["workflow"]["actions"]
        gates = {item["gate"]: False for item in actions}
        gates.update({"pool_complete": True, "strict_pool_validation": True})
        for item in actions:
            expected = item["action"]
            if item.get("requires_training_allowed"):
                gates["training_allowed"] = True
            self.assertEqual(supervisor.select_workflow_action(self.policy, gates), expected)
            gates[item["gate"]] = True
            if item["gate"] == "geometry_split_frozen":
                gates["training_allowed"] = True
        self.assertIsNone(supervisor.select_workflow_action(self.policy, gates))

    def test_reference_and_replacement_precede_new_joint_gate(self):
        gates = {item["gate"]: False for item in self.policy["workflow"]["actions"]}
        gates.update({
            "pool_complete": True,
            "strict_pool_validation": True,
            "pool_manifest_frozen": True,
            "d65_colorimetry": True,
        })
        self.assertEqual(supervisor.select_workflow_action(self.policy, gates), "reference_resolution")
        gates["reference_resolution"] = True
        self.assertEqual(supervisor.select_workflow_action(self.policy, gates), "replacement_pool_generation")
        gates["replacement_pool_ready"] = True
        self.assertEqual(supervisor.select_workflow_action(self.policy, gates), "joint_numerical_convergence")

    def test_post_activation_actions_use_only_v2_protocol_bound_runners(self):
        actions = {
            item["action"]: item for item in self.policy["workflow"]["actions"]
        }
        self.assertEqual(
            actions["reference_resolution"]["evidence_version"],
            "paper2-reference-holdout-audit-v1",
        )
        self.assertIn(
            "launch_reference_resolution_holdout.py",
            actions["reference_resolution"]["runner"],
        )
        self.assertIn(
            "audit_reference_resolution_holdout.py",
            actions["reference_resolution"]["auditor"],
        )
        self.assertEqual(
            actions["joint_numerical_convergence"]["evidence_version"],
            "paper2-joint-convergence-v2",
        )
        self.assertIn(
            "run_joint_convergence_v2.py",
            actions["joint_numerical_convergence"]["runner"],
        )
        self.assertEqual(
            actions["cross_solver_spectrum_validation"]["evidence_version"],
            "paper2-cross-solver-v2",
        )
        self.assertIn(
            "run_cross_solver_validation_v2.py",
            actions["cross_solver_spectrum_validation"]["runner"],
        )
        joint_instruction = supervisor.build_instruction(
            "joint_numerical_convergence", self.policy
        )
        reference_instruction = supervisor.build_instruction(
            "reference_resolution", self.policy
        )
        cross_instruction = supervisor.build_instruction(
            "cross_solver_spectrum_validation", self.policy
        )
        self.assertIn("run_joint_convergence_v2.py", joint_instruction)
        self.assertIn("activated replacement pool", joint_instruction)
        self.assertIn("launch_reference_resolution_holdout.py", reference_instruction)
        self.assertIn("run_cross_solver_validation_v2.py", cross_instruction)
        self.assertIn("active pool", cross_instruction)

    def test_replacement_pool_is_auditor_activated(self):
        instruction = supervisor.build_instruction("replacement_pool_generation", self.policy)
        self.assertIn("do not register the gate", instruction)
        self.assertIn("failure_class=scientific", instruction)
        self.assertIn("do not", instruction.lower())

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
        self.assertEqual(second["effective_status"], "completed")
        self.assertEqual(second["controller_status"], "running")
        self.assertEqual(second["pipeline_status"], "running")
        self.assertFalse(second["pipeline_complete"])
        self.assertEqual(second["next_action"], request["action"])
        audit = supervisor.load_json(supervisor.AUDIT_RESULT)
        plan = supervisor.load_json(supervisor.NEXT_PLAN)
        self.assertEqual(audit["pipeline_status"], "running")
        self.assertFalse(audit["pipeline_complete"])
        self.assertEqual(plan["recommended_next"], request["action"])

    def test_nonterminal_dispatch_survives_workflow_reordering(self):
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        active = {
            "schema_version": 1,
            "protocol_version": 2,
            "request_id": supervisor.make_dispatch_id(
                "paper2_pipeline", "joint_numerical_convergence", pool_sha
            ),
            "target_thread_id": self.policy["executor_thread_id"],
            "stage": "paper2_pipeline",
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "attempt": 1,
            "max_attempts": 3,
            "created_at": supervisor.now_iso(),
            "updated_at": supervisor.now_iso(),
            "ack_required": True,
            "payload": {"pool": "pool.pkl", "pool_sha256": pool_sha},
            "instruction": "frozen active request",
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, active)
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, {
            "request_id": active["request_id"],
            "attempt": 1,
            "status": "running",
            "observed_at": supervisor.now_iso(),
            "lease_expires_at": (
                datetime.now().astimezone() + timedelta(hours=1)
            ).isoformat(timespec="seconds"),
        })
        result = supervisor.update_dispatch(
            "reference_resolution", self.policy, {"pool": {"sha256": "NEW-POOL"}}
        )
        self.assertEqual(result["request_id"], active["request_id"])
        self.assertEqual(result["action"], "joint_numerical_convergence")
        self.assertEqual(result["status"], "in_progress")
        self.assertEqual(result["payload"]["pool_sha256"], pool_sha)
        self.assertEqual(result["instruction"], "frozen active request")

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

    def test_recent_checkpoint_gives_dead_worker_a_finalization_grace(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        checkpoint = self.root / "checkpoint.pkl"
        checkpoint.write_bytes(b"completed checkpoint")
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
                "checkpoint_path": "checkpoint.pkl",
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["status"], "in_progress")
        self.assertEqual(second["dispatch"]["attempt"], 1)
        self.assertIn("finalization_grace_until", second["dispatch"])

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
        failed_id = second["dispatch"]["request_id"]
        failed_updated_at = second["dispatch"]["updated_at"]
        with patch.object(supervisor, "pid_alive", return_value=False):
            third = supervisor.evaluate_once(self.policy)
        self.assertEqual(third["dispatch"]["request_id"], failed_id)
        self.assertEqual(third["dispatch"]["status"], "failed")
        self.assertEqual(third["dispatch"]["updated_at"], failed_updated_at)
        self.assertEqual(third["next_action"], "stop_and_report")

    def test_failed_gate_cannot_advance_without_evidence_backed_strategy(self):
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        failed = {
            "request_id": "failed-joint-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "attempt": 1,
            "max_attempts": 3,
            "terminal_failure": True,
            "payload": {"pool": "pool.pkl", "pool_sha256": pool_sha},
            "instruction": "frozen failed request",
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, failed)
        held = supervisor.update_dispatch(
            "reference_resolution", self.policy, {"pool": {"sha256": pool_sha}}
        )
        self.assertEqual(held, failed)

        evidence = self.root / "repair.json"
        evidence.write_text('{"passed": true}\n', encoding="ascii")
        self.policy["strategy_override"] = {
            "enabled": True,
            "decision": "retry_same_gate",
            "revision": 2,
            "action": "reference_resolution",
            "based_on_request_id": failed["request_id"],
            "instruction_append": "Launch only the independently audited holdout.",
            "evidence": [
                {"path": "repair.json", "sha256": supervisor.file_digest(evidence)}
            ],
        }
        advanced = supervisor.update_dispatch(
            "reference_resolution", self.policy, {"pool": {"sha256": pool_sha}}
        )
        self.assertNotEqual(advanced["request_id"], failed["request_id"])
        self.assertEqual(advanced["action"], "reference_resolution")
        self.assertEqual(advanced["status"], "pending")
        self.assertEqual(advanced["strategy_revision"], 2)

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
        self.assertEqual(plan["recovery_owner"], "independent_auditor")
        self.assertEqual(plan["next_action"], "diagnose_repair_and_replan")
        self.assertFalse(plan["user_intervention_required"])
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

    def test_evidence_backed_strategy_can_retry_after_transient_budget_exhausted(self):
        repair = self.root / "repair_evidence.json"
        repair.write_text('{"passed": true}\n', encoding="ascii")
        failed = {
            "request_id": "exhausted-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "attempt": 3,
            "max_attempts": 3,
            "terminal_failure": False,
        }
        self.policy["strategy_override"] = {
            "enabled": True,
            "decision": "retry_same_gate",
            "revision": 2,
            "action": failed["action"],
            "based_on_request_id": failed["request_id"],
            "instruction_append": "Retry only with the validated permanent repair.",
            "evidence": [
                {"path": "repair_evidence.json", "sha256": supervisor.file_digest(repair)}
            ],
        }
        strategy = supervisor.strategy_override(failed["action"], self.policy, failed)
        self.assertIsNotNone(strategy)

    def test_active_strategy_request_freezes_evidence_and_instruction(self):
        repair = self.root / "repair_evidence.json"
        repair.write_text('{"passed": true, "revision": 1}\n', encoding="ascii")
        failed = {
            "request_id": "failed-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "attempt": 1,
            "max_attempts": 3,
            "terminal_failure": True,
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, failed)
        self.policy["strategy_override"] = {
            "enabled": True,
            "decision": "retry_same_gate",
            "revision": 1,
            "action": failed["action"],
            "based_on_request_id": failed["request_id"],
            "instruction_append": "Use only the hash-bound repair evidence.",
            "evidence": [
                {"path": "repair_evidence.json", "sha256": supervisor.file_digest(repair)}
            ],
        }
        audit = {"pool": {"sha256": "ABC"}}
        first = supervisor.update_dispatch(failed["action"], self.policy, audit)
        first_id = first["request_id"]
        first_instruction = first["instruction"]
        first_evidence = copy.deepcopy(first["strategy_evidence"])

        repair.write_text('{"passed": true, "revision": 2}\n', encoding="ascii")
        new_hash = supervisor.file_digest(repair)
        self.policy["strategy_override"]["evidence"][0]["sha256"] = new_hash
        second = supervisor.update_dispatch(failed["action"], self.policy, audit)
        self.assertEqual(second["request_id"], first_id)
        self.assertEqual(second["strategy_evidence"], first_evidence)
        self.assertNotEqual(second["strategy_evidence"][0]["sha256"], new_hash)
        self.assertEqual(second["instruction"], first_instruction)


if __name__ == "__main__":
    unittest.main()
