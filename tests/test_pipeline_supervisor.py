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


class GatePayloadVerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_root = supervisor.ROOT
        self.old_policy = supervisor.POLICY
        self.old_state = supervisor.STATE
        self.old_dispatch = supervisor.DISPATCH_REQUEST
        self.old_executor_ack = supervisor.EXECUTOR_ACK
        supervisor.ROOT = self.root
        supervisor.STATE = self.root / ".state"
        supervisor.STATE.mkdir()
        supervisor.POLICY = self.root / "pipeline_policy.json"
        supervisor.DISPATCH_REQUEST = supervisor.STATE / "dispatch_request.json"
        supervisor.EXECUTOR_ACK = supervisor.STATE / "executor_ack.json"
        self.policy = {
            "schema_version": 1,
            "protected_files": [],
            "immutable_assets": [],
        }
        supervisor.atomic_json(supervisor.POLICY, self.policy)

    def tearDown(self):
        supervisor.ROOT = self.old_root
        supervisor.POLICY = self.old_policy
        supervisor.STATE = self.old_state
        supervisor.DISPATCH_REQUEST = self.old_dispatch
        supervisor.EXECUTOR_ACK = self.old_executor_ack
        self.tmp.cleanup()

    def binding(self, name: str, content: bytes = b"bound evidence\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": name.replace("\\", "/"), "sha256": supervisor.file_digest(path)}

    def json_binding(self, name: str, payload: dict):
        return self.binding(
            name,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii"),
        )

    def pickle_binding(self, name: str, payload):
        return self.binding(name, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    def protected_snapshot(self):
        item = self.binding("paper1.locked", b"immutable paper 1\n")
        md5 = supervisor.file_digest(self.root / item["path"], "md5")
        self.policy["protected_files"] = [{"path": item["path"], "md5": md5}]
        supervisor.atomic_json(supervisor.POLICY, self.policy)
        return [{
            "path": item["path"],
            "expected_md5": md5,
            "actual_md5": md5,
            "passed": True,
        }]

    def runtime_hashes(self, names):
        return {
            name: self.binding(name, f"# {name}\n".encode("ascii"))["sha256"]
            for name in names
        }

    def authorize(self, action: str, request=None, status="in_progress"):
        request = request or {"request_id": f"{action}-request", "attempt": 1}
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, {
            "action": action,
            "status": status,
            **request,
        })
        return request

    def audited_payload(
        self,
        worker: dict,
        *,
        action: str,
        audit_version: str,
        independent: dict,
    ):
        worker = copy.deepcopy(worker)
        worker.setdefault("schema_version", 1)
        worker["request"] = self.authorize(action)
        worker_binding = self.json_binding(f"state/{action}-worker.json", worker)
        audit = copy.deepcopy(worker)
        audit["evidence_version"] = audit_version
        audit["worker_evidence"] = worker_binding
        audit["independent_reproduction"] = True
        audit.update(copy.deepcopy(independent))
        audit["auditor_runtime_hashes"] = self.runtime_hashes(
            supervisor.AUDITOR_RUNTIME_PATHS[action]
        )
        return audit

    def authorize_failed_worker(self, audit: dict):
        request = copy.deepcopy(audit["request"])
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, {
            "action": "replacement_pool_generation",
            "status": "failed",
            "terminal_failure": True,
            "failure_class": "scientific",
            **request,
        })
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, {
            "status": "failed",
            "failure_class": "scientific",
            "evidence": [copy.deepcopy(audit["worker_evidence"])],
            **request,
        })

    def comparison_set(self, count: int):
        names = (
            "order_365x768_to_450x768_0p5nm",
            "grid_450x512_to_450x768_0p5nm",
            "corner_365x512_to_450x768_0p5nm",
            "spectral_450x768_1nm_to_0p5nm",
            "frozen_candidate_to_final_reference",
        )
        start_index = 8 if count == 24 else 0
        rows = [
            {"geometry_index": index, "pol": pol, "dE00": 0.0}
            for index in range(start_index, start_index + count)
            for pol in ("p", "s")
        ]
        return {
            name: {
                "count": count,
                "mean": 0.0,
                "max": 0.0,
                "mean_lt_1_15": True,
                "all_lt_2_3": True,
                "passed": True,
                "joint_max_by_geometry": [0.0] * count,
                "rows": copy.deepcopy(rows),
            }
            for name in names
        }

    def test_minimal_forged_payloads_are_rejected_by_every_registered_gate(self):
        bound = self.binding("state/decoy.bin")
        pool = {"sha256": "A" * 64, "passed": True}
        cases = (
            (supervisor.verify_reference_resolution_gate, {
                "schema_version": 1,
                "evidence_version": "paper2-reference-holdout-audit-v1",
                "protocol_revision": "v2_bound_holdout",
                "classification": "reference_holdout_passed",
                "passed": True,
                "production_reference_approved": True,
                "final_reference": copy.deepcopy(supervisor.REFERENCE_HOLDOUT_FINAL_REFERENCE),
                "training_allowed": False,
                "checks": {},
                "sources": {"decoy": bound},
            }),
            (supervisor.verify_replacement_pool_gate, {
                "schema_version": 1,
                "passed": True,
                "training_allowed": False,
                "pool_sha256": pool["sha256"],
                "pool_spec": {"path": bound["path"], "expected_records": 1},
                "approved_protocol": bound,
                "checkpoint": bound,
                "runtime_hashes": {bound["path"]: bound["sha256"]},
                "audit": {},
            }),
            (supervisor.verify_joint_v2_gate, {
                "passed": True,
                "classification": "passed",
                "training_allowed": False,
                "pool_sha256": pool["sha256"],
                "checks": {
                    "active_pool_strict_audit": True,
                    "paper1_and_legacy_assets_unchanged": True,
                    "replacement_vs_reference": True,
                },
                "thresholds": {
                    "mean_joint_dE00_lt": 1.15,
                    "all_joint_dE00_lt": 2.3,
                    "pointwise_conservation_lte": 1e-6,
                    "stored_label_atol": 1e-10,
                },
                "evaluation": {"passed": True, "checks": {}, "joint_dE00": {"count": 32}},
            }),
            (supervisor.verify_cross_solver_v2_gate, {
                "passed": True,
                "classification": "passed",
                "training_allowed": False,
                "pool_sha256": pool["sha256"],
                "checks": {
                    "controls": True,
                    "matched_results": True,
                    "runtime_hashes_verified": True,
                    "paper1_and_legacy_assets_unchanged": True,
                },
                "controls": {"passed": True, "checks": {}},
                "evaluation": {"passed": True, "checks": {}},
            }),
        )
        for verifier, payload in cases:
            with self.subTest(verifier=verifier.__name__):
                self.assertFalse(verifier(payload, pool)[0])

    def test_auditor_envelope_rejects_replay_worker_tamper_and_runtime_tamper(self):
        worker = {
            "schema_version": 1,
            "evidence_version": "paper2-joint-convergence-v2",
            "request": {"request_id": "stable-request", "attempt": 1},
            "passed": True,
        }
        worker_binding = self.json_binding("state/joint-worker.json", worker)
        active_request = {"request_id": "stable-request", "attempt": 2}
        self.authorize("joint_numerical_convergence", active_request)
        payload = copy.deepcopy(worker)
        payload.update({
            "evidence_version": "paper2-joint-convergence-audit-v1",
            "request": active_request,
            "worker_evidence": worker_binding,
            "independent_reproduction": True,
            "independent_evaluation": {"passed": True},
            "auditor_runtime_hashes": self.runtime_hashes(
                supervisor.AUDITOR_RUNTIME_PATHS["joint_numerical_convergence"]
            ),
        })
        verified, error = supervisor.audited_worker_payload(
            payload,
            action="joint_numerical_convergence",
            audit_version="paper2-joint-convergence-audit-v1",
            worker_version="paper2-joint-convergence-v2",
            audit_fields={"independent_evaluation"},
            auditor_runtime_paths=supervisor.AUDITOR_RUNTIME_PATHS[
                "joint_numerical_convergence"
            ],
        )
        self.assertEqual(verified, worker)
        self.assertIsNone(error)

        replay = copy.deepcopy(payload)
        replay["request"] = {"request_id": "old-request", "attempt": 1}
        self.assertIsNotNone(supervisor.audited_worker_payload(
            replay,
            action="joint_numerical_convergence",
            audit_version="paper2-joint-convergence-audit-v1",
            worker_version="paper2-joint-convergence-v2",
            audit_fields={"independent_evaluation"},
            auditor_runtime_paths=supervisor.AUDITOR_RUNTIME_PATHS[
                "joint_numerical_convergence"
            ],
        )[1])

        worker_path = self.root / worker_binding["path"]
        original = worker_path.read_bytes()
        worker_path.write_bytes(b"{}\n")
        self.assertIsNotNone(supervisor.audited_worker_payload(
            payload,
            action="joint_numerical_convergence",
            audit_version="paper2-joint-convergence-audit-v1",
            worker_version="paper2-joint-convergence-v2",
            audit_fields={"independent_evaluation"},
            auditor_runtime_paths=supervisor.AUDITOR_RUNTIME_PATHS[
                "joint_numerical_convergence"
            ],
        )[1])
        worker_path.write_bytes(original)

        runtime_tampered = copy.deepcopy(payload)
        first_runtime = next(iter(runtime_tampered["auditor_runtime_hashes"]))
        runtime_tampered["auditor_runtime_hashes"][first_runtime] = "0" * 64
        self.assertIsNotNone(supervisor.audited_worker_payload(
            runtime_tampered,
            action="joint_numerical_convergence",
            audit_version="paper2-joint-convergence-audit-v1",
            worker_version="paper2-joint-convergence-v2",
            audit_fields={"independent_evaluation"},
            auditor_runtime_paths=supervisor.AUDITOR_RUNTIME_PATHS[
                "joint_numerical_convergence"
            ],
        )[1])

    def test_reference_resolution_verifier_accepts_only_frozen_holdout_contract(self):
        candidate = {
            "requested_nG": 365,
            "Nxy": 512,
            "wavelength_step_nm": 1.0,
            "passed": True,
        }
        thresholds = {
            "mean_joint_dE00_lt": 1.15,
            "all_joint_dE00_lt": 2.3,
            "pointwise_conservation_lte": 1e-6,
        }
        frozen_pool = self.binding("data/frozen-pool.pkl")
        sources = {}
        for name in (
            "source_v2_plan", "source_v2_checkpoint", "source_v2_worker_evidence",
            "source_v2_independent_audit", "source_base_checkpoint",
            "holdout_evidence", "holdout_checkpoint",
        ):
            sources[name] = self.binding(f"state/{name}.bin")
        plan_payload = {
            "final_reference": copy.deepcopy(supervisor.REFERENCE_HOLDOUT_FINAL_REFERENCE),
            "frozen_candidate": candidate,
            "thresholds": thresholds,
            "pool": frozen_pool,
            "primary_gate_population": "24_new_holdout_geometries_only",
            "combined_32_population_scope": "supplemental_reporting_only",
        }
        for name in (
            "source_v2_plan", "source_v2_checkpoint", "source_v2_worker_evidence",
            "source_v2_independent_audit", "source_base_checkpoint",
        ):
            plan_payload[name] = sources[name]
        plan = self.json_binding("state/holdout-plan.json", plan_payload)
        sources = {"plan": plan, **sources}
        payload = {
            "schema_version": 1,
            "evidence_version": "paper2-reference-holdout-audit-v1",
            "protocol_revision": "v2_bound_holdout",
            "passed": True,
            "production_reference_approved": True,
            "classification": "reference_holdout_passed",
            "request": {"request_id": "reference-request", "attempt": 1},
            "final_reference": copy.deepcopy(supervisor.REFERENCE_HOLDOUT_FINAL_REFERENCE),
            "training_allowed": False,
            "primary_gate_population": "24_new_holdout_geometries_only",
            "combined_32_population_scope": "supplemental_reporting_only",
            "checks": {
                "policy_integrity": True,
                "paper1_and_legacy_assets_unchanged": True,
                "v2_plan_and_all_source_hashes_verified": True,
                "candidate_independently_refrozen_on_initial_eight": True,
                "holdout_did_not_reselect_candidate": True,
                "exact_extension_task_set": True,
                "worker_evidence_exactly_reproduced": True,
                "production_reference_approved": True,
            },
            "thresholds": thresholds,
            "approved_protocol_candidate": candidate,
            "pool_sha256": frozen_pool["sha256"],
            "sources": sources,
            "worker_evidence": sources["holdout_evidence"],
            "independent_reproduction": True,
            "auditor_runtime_hashes": self.runtime_hashes(
                supervisor.AUDITOR_RUNTIME_PATHS["reference_resolution"]
            ),
            "independent_holdout_comparisons": self.comparison_set(24),
            "combined_32_supplemental_comparisons": self.comparison_set(32),
            "protected_files": self.protected_snapshot(),
        }
        self.authorize("reference_resolution", payload["request"])
        self.assertEqual(supervisor.verify_reference_resolution_gate(payload, {}), (True, None))

        tampered_rows = copy.deepcopy(payload)
        tampered_rows["independent_holdout_comparisons"][
            "frozen_candidate_to_final_reference"
        ]["rows"][0]["dE00"] = 0.5
        self.assertFalse(supervisor.verify_reference_resolution_gate(tampered_rows, {})[0])

        tampered = copy.deepcopy(payload)
        tampered["checks"]["undeclared_check"] = True
        self.assertFalse(supervisor.verify_reference_resolution_gate(tampered, {})[0])

    def test_replacement_pool_verifier_recomputes_all_disk_bindings(self):
        pool_binding = self.binding("data/replacement/pool.pkl", b"replacement-pool")
        reference_payload = {
            "evidence_version": "paper2-reference-holdout-audit-v1",
            "protocol_revision": "v2_bound_holdout",
            "classification": "reference_holdout_passed",
            "final_reference": copy.deepcopy(supervisor.REFERENCE_HOLDOUT_FINAL_REFERENCE),
            "passed": True,
            "production_reference_approved": True,
        }
        reference = self.json_binding("state/reference-gate.json", reference_payload)
        pool_spec = {
            "path": pool_binding["path"],
            "expected_records": 6000,
            "pointwise_conservation_tolerance": 1e-6,
        }
        protocol = self.json_binding("state/replacement-protocol.json", {
            "schema_version": 1,
            "evidence_version": "paper2-replacement-protocol-v1",
            "protocol_revision": "v2_bound_holdout",
            "approved": True,
            "automatic_launch_authorized": True,
            "pool_spec": pool_spec,
            "source_reference_gate": reference,
        })
        checkpoint = self.binding("state/replacement-checkpoint.sqlite")
        runtime = self.runtime_hashes({
            "rcwa_batch.py", "paper2_colorimetry_fine.py", "scripts/run_replacement_pool.py"
        })
        activation_id = hashlib.sha256(
            f"{protocol['sha256']}|{pool_binding['sha256']}".encode("ascii")
        ).hexdigest()[:24]
        payload = {
            "schema_version": 1,
            "evidence_version": "paper2-replacement-pool-v1",
            "passed": True,
            "training_allowed": False,
            "pool_sha256": pool_binding["sha256"],
            "pool_md5": supervisor.file_digest(self.root / pool_binding["path"], "md5"),
            "size_bytes": (self.root / pool_binding["path"]).stat().st_size,
            "pool_spec": pool_spec,
            "approved_protocol": protocol,
            "reference_gate_evidence": reference,
            "activation_id": activation_id,
            "audit": {
                "records": 6000,
                "expected_records": 6000,
                "geometries": 3000,
                "complete_pairs": 3000,
                "duplicate_keys": 0,
                "R_plus_T_mean": 1.0,
                "R_plus_T_min": 1.0,
                "R_plus_T_max": 1.0,
                "pointwise_conservation_error_max": 1e-9,
                "R_min": 0.0,
                "R_max": 1.0,
                "T_min": 0.0,
                "T_max": 1.0,
            },
            "checkpoint": checkpoint | {"failure_events": 0},
            "runtime_hashes": runtime,
            "protected_files": self.protected_snapshot(),
        }
        payload = self.audited_payload(
            payload,
            action="replacement_pool_generation",
            audit_version="paper2-replacement-pool-audit-v1",
            independent={"independent_audit": copy.deepcopy(payload["audit"])},
        )
        self.authorize_failed_worker(payload)
        pool = {"sha256": pool_binding["sha256"], "passed": True, **payload["audit"]}
        self.assertEqual(supervisor.verify_replacement_pool_gate(payload, pool), (True, None))

        missing_ack_binding = supervisor.load_json(supervisor.EXECUTOR_ACK)
        missing_ack_binding["evidence"] = [self.binding("state/other-worker.json")]
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, missing_ack_binding)
        self.assertFalse(supervisor.verify_replacement_pool_gate(payload, pool)[0])

        retried = copy.deepcopy(payload)
        retried["request"]["attempt"] = 2
        self.authorize_failed_worker(retried)
        self.assertEqual(supervisor.verify_replacement_pool_gate(retried, pool), (True, None))

        tampered = copy.deepcopy(payload)
        tampered["checkpoint"]["sha256"] = "0" * 64
        self.assertFalse(supervisor.verify_replacement_pool_gate(tampered, pool)[0])

    def test_joint_v2_verifier_enforces_exact_thresholds_and_bindings(self):
        pool_sha = "A" * 64
        pool_spec = {"path": "data/replacement/pool.pkl", "expected_records": 6000}
        protocol = self.json_binding("state/approved-protocol.json", {
            "schema_version": 1,
            "evidence_version": "paper2-replacement-protocol-v1",
            "protocol_revision": "v2_bound_holdout",
            "approved": True,
            "automatic_launch_authorized": True,
            "pool_spec": pool_spec,
        })
        active = self.json_binding("state/active-pool.json", {
            "schema_version": 1,
            "evidence_version": "paper2-active-pool-v1",
            "active": True,
            "training_allowed": False,
            "pool_sha256": pool_sha,
            "approved_protocol": protocol,
            "pool_spec": pool_spec,
        })
        runtime = self.runtime_hashes({
            "paper2_colorimetry_fine.py", "scripts/run_joint_convergence_v2.py"
        })
        raw_hashes = {
            "base_checkpoint_sha256": "1" * 64,
            "budget_v2_checkpoint_sha256": "2" * 64,
            "holdout_checkpoint_sha256": "3" * 64,
        }
        reference = self.json_binding("state/reference-gate.json", {
            "evidence_version": "paper2-reference-holdout-audit-v1",
            "protocol_revision": "v2_bound_holdout",
            "classification": "reference_holdout_passed",
            "final_reference": copy.deepcopy(supervisor.REFERENCE_HOLDOUT_FINAL_REFERENCE),
            "passed": True,
            "production_reference_approved": True,
            "sources": {
                "source_base_checkpoint": {"sha256": raw_hashes["base_checkpoint_sha256"]},
                "source_v2_checkpoint": {"sha256": raw_hashes["budget_v2_checkpoint_sha256"]},
                "holdout_checkpoint": {"sha256": raw_hashes["holdout_checkpoint_sha256"]},
            },
        })
        payload = {
            "schema_version": 1,
            "evidence_version": "paper2-joint-convergence-v2",
            "passed": True,
            "classification": "passed",
            "training_allowed": False,
            "pool_sha256": pool_sha,
            "checks": {
                "active_pool_strict_audit": True,
                "paper1_and_legacy_assets_unchanged": True,
                "replacement_vs_reference": True,
            },
            "thresholds": {
                "mean_joint_dE00_lt": 1.15,
                "all_joint_dE00_lt": 2.3,
                "pointwise_conservation_lte": 1e-6,
                "stored_label_atol": 1e-10,
            },
            "evaluation": {
                "passed": True,
                "checks": {
                    "exact_32_complete_p_s_geometries": True,
                    "derived_labels_exact": True,
                    "pool_conservation": True,
                    "reference_conservation": True,
                    "mean_joint_dE00": True,
                    "all_joint_dE00": True,
                },
                "joint_dE00": {
                    "count": 32,
                    "mean": 0.0,
                    "median": 0.0,
                    "max": 0.0,
                    "values": [0.0] * 32,
                },
                "pointwise_conservation_error_max": 0.0,
                "reference_pointwise_conservation_error_max": 0.0,
                "missing": [],
                "label_failures": [],
            },
            "active_pool": active,
            "approved_protocol": protocol,
            "reference_gate": reference,
            "reference_raw_spectra": raw_hashes,
            "runtime_hashes": runtime,
            "protected_files": self.protected_snapshot(),
        }
        payload = self.audited_payload(
            payload,
            action="joint_numerical_convergence",
            audit_version="paper2-joint-convergence-audit-v1",
            independent={"independent_evaluation": copy.deepcopy(payload["evaluation"])},
        )
        pool = {"sha256": pool_sha}
        self.assertEqual(supervisor.verify_joint_v2_gate(payload, pool), (True, None))

        tampered = copy.deepcopy(payload)
        tampered["thresholds"]["stored_label_atol"] = 1e-9
        self.assertFalse(supervisor.verify_joint_v2_gate(tampered, pool)[0])

    def test_cross_solver_verifier_enforces_protocol_and_checkpoint_binding(self):
        pool_sha = "B" * 64
        production = {"nG_requested": 365, "nG_retained": 361, "Nxy": 512}
        stress = [
            {"name": "order_axis", "nG_requested": 450, "Nxy": 512},
            {"name": "grid_axis", "nG_requested": 365, "Nxy": 768},
            {"name": "higher_corner", "nG_requested": 450, "Nxy": 768},
        ]
        pool_spec = {"path": "data/replacement/pool.pkl", "expected_records": 6000}
        protocol = self.json_binding("state/approved-protocol.json", {
            "schema_version": 1,
            "evidence_version": "paper2-replacement-protocol-v1",
            "protocol_revision": "v2_bound_holdout",
            "approved": True,
            "automatic_launch_authorized": True,
            "pool_spec": pool_spec,
        })
        active = self.json_binding("state/active-pool.json", {
            "schema_version": 1,
            "evidence_version": "paper2-active-pool-v1",
            "active": True,
            "training_allowed": False,
            "pool_sha256": pool_sha,
            "approved_protocol": protocol,
            "pool_spec": pool_spec,
        })
        runtime = self.runtime_hashes({
            "rcwa_batch.py", "paper2_colorimetry_fine.py",
            "scripts/run_cross_solver_validation.py", "scripts/run_cross_solver_validation_v2.py",
        })
        control_checks = {
            f"{solver}_{metric}": True
            for solver in ("grcwa", "thirdparty")
            for metric in (
                "fresnel_error", "empty_energy_error", "circle_max_difference",
                "rotation_max_difference",
            )
        }
        selected = [
            {"L": 200.0 + index, "W": 150.0, "H": 300.0, "P": 400.0, "stress": index < 4}
            for index in range(12)
        ]
        thresholds = {
            "per_spectrum_R_T_rmse_lte": 0.05,
            "mean_spectrum_R_T_rmse_lte": 0.03,
            "mean_joint_dE00_lt": 1.15,
            "per_geometry_joint_dE00_lt": 2.3,
            "energy_error_lte": 1e-6,
            "analytic_and_symmetry_error_lte": 1e-7,
        }
        wavelength = np.arange(380.0, 785.0, 5.0)
        results = {}
        for index in range(12):
            modes = ["production"] + ([item["name"] for item in stress] if index < 4 else [])
            for pol in ("p", "s"):
                for mode in modes:
                    task_id = f"crossv2-g{index:02d}-{pol}-{mode}"
                    results[task_id] = {
                        "id": task_id,
                        "status": "ok",
                        "geometry_index": index,
                        "pol": pol,
                        "mode": mode,
                        "wavelength_nm": wavelength,
                        "grcwa_R": np.full(wavelength.size, 0.2),
                        "grcwa_T": np.full(wavelength.size, 0.8),
                        "thirdparty_R": np.full(wavelength.size, 0.2),
                        "thirdparty_T": np.full(wavelength.size, 0.8),
                    }
        checkpoint_meta = {
            "version": "paper2-cross-solver-v2",
            "pool_sha256": pool_sha,
            "approved_protocol_sha256": protocol["sha256"],
            "selected_geometries": selected,
            "production": production,
            "stress_configs": stress,
            "expected_tasks": len(results),
            "runtime_hashes": runtime,
            "thresholds": thresholds,
        }
        checkpoint = self.pickle_binding(
            "state/cross-checkpoint.pkl", {"meta": checkpoint_meta, "results": results}
        ) | {"tasks": len(results)}
        payload = {
            "schema_version": 1,
            "evidence_version": "paper2-cross-solver-v2",
            "passed": True,
            "classification": "passed",
            "training_allowed": False,
            "pool_sha256": pool_sha,
            "checks": {
                "controls": True,
                "matched_results": True,
                "runtime_hashes_verified": True,
                "paper1_and_legacy_assets_unchanged": True,
            },
            "controls": {"passed": True, "checks": control_checks},
            "evaluation": {
                "passed": True,
                "classification": "passed",
                "checks": {
                    "no_task_failures": True,
                    "production_cross_solver": True,
                    "all_stress_cross_solver": True,
                    "both_solvers_converged": True,
                },
                "failures": [],
            },
            "protocol": {
                "production": production,
                "stress_configs": stress,
                "geometry_count": 12,
                "stress_geometry_count": 4,
                "polarizations": ["p", "s"],
                "wavelength_nm": wavelength.tolist(),
                "background": "air",
                "incident": "air",
                "transmission_halfspace": "SiO2",
            },
            "thresholds": thresholds,
            "selected_geometries": selected,
            "raw_checkpoint": checkpoint,
            "active_pool": active,
            "approved_protocol": protocol,
            "runtime_hashes": runtime,
            "protected_files": self.protected_snapshot(),
        }
        payload = self.audited_payload(
            payload,
            action="cross_solver_spectrum_validation",
            audit_version="paper2-cross-solver-audit-v1",
            independent={
                "independent_controls": copy.deepcopy(payload["controls"]),
                "independent_evaluation": copy.deepcopy(payload["evaluation"]),
            },
        )
        pool = {"sha256": pool_sha}
        self.assertEqual(supervisor.verify_cross_solver_v2_gate(payload, pool), (True, None))

        checkpoint_path = self.root / payload["raw_checkpoint"]["path"]
        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint_path.write_bytes(checkpoint_bytes + b"tampered")
        self.assertFalse(supervisor.verify_cross_solver_v2_gate(payload, pool)[0])
        checkpoint_path.write_bytes(checkpoint_bytes)

        tampered = copy.deepcopy(payload)
        tampered["protocol"]["background"] = "substrate"
        self.assertFalse(supervisor.verify_cross_solver_v2_gate(tampered, pool)[0])


class NewGateContractTests(unittest.TestCase):
    def test_geometry_split_verifier_rejects_duplicate_geometry(self):
        assignments = [
            {"geometry_id": f"g-{index}", "split": "train" if index < 8 else "validation" if index == 8 else "test"}
            for index in range(10)
        ]
        payload = {
            "passed": True,
            "classification": "geometry_split_frozen",
            "training_allowed": False,
            "pool_sha256": "A" * 64,
            "split_version": "sha256-ranked-80-10-10-v1",
            "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "checks": {
                "active_pool_hash_verified": True,
                "canonical_axes_verified": True,
                "exact_dual_polarization_pairs": True,
                "stable_geometry_ids_verified": True,
                "geometry_level_no_leakage": True,
                "split_counts_exact": True,
            },
            "geometry_count": 10,
            "record_count": 20,
            "counts": {"train": 8, "validation": 1, "test": 1},
            "assignments": assignments,
            "assignments_sha256": supervisor.json_payload_digest(assignments),
            "runtime_hashes": {},
        }
        with patch.object(
            supervisor, "audited_worker_payload", return_value=({}, None)
        ), patch.object(supervisor, "runtime_hashes_match", return_value=(True, None)):
            self.assertEqual(
                supervisor.verify_geometry_split_gate(payload, {"sha256": "A" * 64}),
                (True, None),
            )
            tampered = copy.deepcopy(payload)
            tampered["assignments"][1]["geometry_id"] = tampered["assignments"][0]["geometry_id"]
            tampered["assignments_sha256"] = supervisor.json_payload_digest(
                tampered["assignments"]
            )
            self.assertFalse(
                supervisor.verify_geometry_split_gate(tampered, {"sha256": "A" * 64})[0]
            )

    def test_circular_verifier_enforces_registered_symmetry_threshold(self):
        geometries = [
            {"control_id": f"c-{index}", "D": 120.0, "H": 200.0, "P": 400.0}
            for index in range(12)
        ]
        metrics = [
            {
                "id": f"c-{index}",
                "valid": True,
                "max_pointwise_conservation_error": 1e-10,
                "polarization_R_max_abs": 1e-10,
                "polarization_T_max_abs": 1e-10,
                "polarization_dE00": 1e-5,
            }
            for index in range(12)
        ]
        payload = {
            "passed": True,
            "classification": "circular_control_passed",
            "training_allowed": False,
            "pool_sha256": "B" * 64,
            "checks": {
                "exact_frozen_geometry_set": True,
                "no_task_failures": True,
                "pointwise_conservation": True,
                "circular_polarization_spectrum_symmetry": True,
                "circular_polarization_color_symmetry": True,
            },
            "protocol": {
                "material": "TiO2",
                "substrate": "SiO2",
                "background": "air",
                "nG_requested": 131,
                "nG_retained": 121,
                "Nxy": 256,
                "wavelength_step_nm": 5.0,
            },
            "thresholds": {
                "pointwise_conservation_lte": 1e-6,
                "polarization_spectrum_max_abs_lte": 1e-7,
                "polarization_dE00_lte": 0.01,
            },
            "selected_geometries": geometries,
            "metrics": metrics,
            "raw_checkpoint": {"tasks": 12},
            "runtime_hashes": {},
        }
        with patch.object(
            supervisor, "audited_worker_payload", return_value=({}, None)
        ), patch.object(supervisor, "bindings_exist", return_value=(True, None)), patch.object(
            supervisor, "runtime_hashes_match", return_value=(True, None)
        ):
            self.assertEqual(
                supervisor.verify_circular_control_gate(payload, {"sha256": "B" * 64}),
                (True, None),
            )
            payload["metrics"][0]["polarization_R_max_abs"] = 1e-6
            self.assertFalse(
                supervisor.verify_circular_control_gate(payload, {"sha256": "B" * 64})[0]
            )


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

    def write_d65_gate(self, *, evidence_version="paper2-d65-v1"):
        implementation = self.root / "paper2_colorimetry.py"
        tests = self.root / "test_paper2_colorimetry.py"
        implementation.write_text("# frozen D65 implementation\n", encoding="ascii")
        tests.write_text("# frozen D65 tests\n", encoding="ascii")
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        evidence = self.root / "d65_evidence.json"
        supervisor.atomic_json(evidence, {
            "schema_version": 1,
            "evidence_version": evidence_version,
            "evidence_revision": 2,
            "passed": True,
            "checks": {
                "pool_records_6000": True,
                "pool_grid_exact": True,
                "perfect_reflector_lab_neutral": True,
                "perfect_reflector_d65_xy": True,
                "black_reflector_lab_zero": True,
                "lab_source_unclipped_xyz": True,
                "srgb_display_only": True,
            },
            "reference_cases": {
                "perfect_reflector": {"lab": [100.0, 0.0, 0.0]},
                "black_reflector": {"lab": [0.0, 0.0, 0.0]},
                "white_xy": [0.3127, 0.3290],
            },
            "derived_label_provenance": {
                "lab_source": "direct_unclipped_xyz",
                "srgb_role": "display_only_clipped",
            },
            "pool": {"path": "pool.pkl", "sha256": pool_sha, "records": 2},
            "implementation": {
                "path": "paper2_colorimetry.py",
                "sha256": supervisor.file_digest(implementation),
            },
            "tests": {
                "path": "test_paper2_colorimetry.py",
                "sha256": supervisor.file_digest(tests),
            },
            "legacy_path_modified": False,
        })
        supervisor.atomic_json(supervisor.GATE_STATE, {
            "schema_version": 1,
            "gates": {
                "d65_colorimetry": {
                    "passed": True,
                    "checked_at": supervisor.now_iso(),
                    "evidence": [
                        {"path": "d65_evidence.json", "sha256": supervisor.file_digest(evidence)}
                    ],
                }
            },
        })
        return evidence, {"passed": True, "sha256": pool_sha, "records": 2}

    def test_stale_running_is_reconciled_from_artifact(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            result = supervisor.evaluate_once(self.policy)
        self.assertEqual(result["effective_status"], "completed")
        self.assertTrue(result["status_reconciled"])
        self.assertEqual(result["next_action"], "pool_validation")
        self.assertEqual(result["controller_status"], "pending")
        self.assertEqual(result["pipeline_status"], "pending")
        self.assertFalse(result["pipeline_complete"])
        self.assertFalse(result["training_allowed"])

    def test_failed_producer_is_never_relabelled_passed(self):
        self.write_status({"status": "failed", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            result = supervisor.evaluate_once(self.policy)
        audit = supervisor.load_json(supervisor.AUDIT_RESULT)
        self.assertEqual(result["controller_status"], "blocked")
        self.assertEqual(result["next_action"], "stop_and_report")
        self.assertFalse(audit["passed"])

    def test_auto_transition_runs_only_for_terminal_scientific_joint_failure(self):
        with patch.object(supervisor.subprocess, "run") as run:
            self.assertIsNone(
                supervisor.run_auto_transition(
                    {"dispatch": {"action": "joint_numerical_convergence", "status": "in_progress"}}
                )
            )
            run.assert_not_called()

        completed = supervisor.subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout='{"status":"advanced","transition":"reference_budget_v2"}\n',
            stderr="",
        )
        controller = {
            "dispatch": {
                "action": "joint_numerical_convergence",
                "status": "failed",
                "terminal_failure": True,
                "failure_class": "scientific",
            }
        }
        with patch.object(supervisor.subprocess, "run", return_value=completed) as run:
            result = supervisor.run_auto_transition(controller)
        self.assertEqual(result["status"], "advanced")
        self.assertIn("paper2_auto_transition.py", run.call_args.args[0][1])

        activation = supervisor.subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout='{"active":true,"pool_sha256":"ABC"}\n',
            stderr="",
        )
        audit_completed = supervisor.subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout='{"passed":true,"pool_sha256":"ABC"}\n',
            stderr="",
        )
        controller["dispatch"]["action"] = "replacement_pool_generation"
        controller["dispatch"]["request_id"] = "replacement-request"
        replacement_audit = supervisor.STATE / "replacement_pool_v1_audit.json"
        replacement_audit.write_text("{}\n", encoding="ascii")
        with patch.object(
            supervisor.subprocess, "run", side_effect=[audit_completed, activation]
        ) as run:
            result = supervisor.run_auto_transition(controller)
        self.assertEqual(result["transition"], "replacement_pool_activation")
        self.assertEqual(result["based_on_request_id"], "replacement-request")
        self.assertEqual(run.call_count, 2)
        self.assertIn("audit_replacement_pool.py", run.call_args_list[0].args[0][1])
        self.assertIn("activate_replacement_pool.py", run.call_args_list[1].args[0][1])

    def test_user_pause_blocks_post_terminal_transition_for_exact_request(self):
        dispatch = {
            "request_id": "paused-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "terminal_failure": True,
            "failure_class": "scientific",
        }
        policy = {
            "operations": {
                "pause_after_request": {
                    "enabled": True,
                    "request_id": "paused-request",
                    "reason": "user_requested_safe_pause",
                    "resume_requires": "explicit_user_authorization",
                }
            }
        }
        with patch.object(supervisor.subprocess, "run") as run:
            result = supervisor.run_auto_transition({"dispatch": dispatch}, policy)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["based_on_request_id"], "paused-request")
        self.assertEqual(result["resume_requires"], "explicit_user_authorization")
        run.assert_not_called()

        policy["operations"]["pause_after_request"]["request_id"] = "other-request"
        completed = supervisor.subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout='{"status":"advanced","transition":"reference_budget_v2"}\n',
            stderr="",
        )
        with patch.object(supervisor.subprocess, "run", return_value=completed) as run:
            result = supervisor.run_auto_transition({"dispatch": dispatch}, policy)
        self.assertEqual(result["status"], "advanced")
        run.assert_called_once()

    def test_auto_transition_arms_only_matching_terminal_integrity_failure(self):
        dispatch = {
            "request_id": "integrity-request",
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "failed",
            "terminal_failure": True,
            "failure_class": "permanent",
        }
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": dispatch["request_id"],
                "attempt": 1,
                "status": "failed",
                "checks": {
                    "finalization_classification": "execution_integrity_failure"
                },
            },
        )
        completed = supervisor.subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=(
                '{"status":"armed","strategy_revision":3,'
                '"target_request_id":"target-request"}\n'
            ),
            stderr="",
        )
        with patch.object(supervisor.subprocess, "run", return_value=completed) as run:
            result = supervisor.run_auto_transition({"dispatch": dispatch})
        self.assertEqual(result["status"], "armed")
        self.assertIn("arm_reference_budget_v2_audit_recovery.py", run.call_args.args[0][1])

        mismatched = supervisor.load_json(supervisor.EXECUTOR_ACK)
        mismatched["request_id"] = "other-request"
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, mismatched)
        with patch.object(supervisor.subprocess, "run") as run:
            self.assertIsNone(supervisor.run_auto_transition({"dispatch": dispatch}))
        run.assert_not_called()

    def test_executor_finalizer_waits_for_live_worker_and_stale_evidence(self):
        dispatch = {
            "request_id": "live-finalizer-request",
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "payload": {"pool_sha256": supervisor.file_digest(self.root / "pool.pkl")},
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, dispatch)
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, {
            "request_id": dispatch["request_id"],
            "attempt": 1,
            "status": "running",
            "worker_pid": 999999,
        })
        with patch.object(supervisor, "pid_alive", return_value=True), patch.object(
            supervisor.subprocess, "run"
        ) as run:
            self.assertIsNone(supervisor.run_executor_finalization(self.policy))
        run.assert_not_called()

        with patch.object(supervisor, "pid_alive", return_value=False):
            result = supervisor.run_executor_finalization(self.policy)
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["reason"], "worker_evidence_missing")

    def test_executor_finalizer_replaces_dead_complete_ack_before_transition(self):
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        dispatch = {
            "request_id": "dead-finalizer-request",
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "payload": {"pool_sha256": pool_sha},
            "strategy_evidence": [
                {"path": ".state/reference_resolution_budget_v2_plan.json"}
            ],
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, dispatch)
        checkpoint = self.root / "checkpoint.pkl"
        checkpoint.write_bytes(b"complete")
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, {
            "request_id": dispatch["request_id"],
            "attempt": 1,
            "status": "running",
            "worker_pid": 999999,
            "checkpoint_path": "checkpoint.pkl",
        })
        evidence = self.root / ".state/reference_resolution_budget_v2.json"
        supervisor.atomic_json(evidence, {
            "request": {"request_id": dispatch["request_id"], "attempt": 1},
        })
        diagnostic = self.root / "finalization-diagnostic.json"
        diagnostic.write_text("{}\n", encoding="ascii")
        terminal = {
            "request_id": dispatch["request_id"],
            "attempt": 1,
            "status": "failed",
            "failure_class": "scientific",
            "checks": {"pool_sha256": pool_sha},
            "evidence": [{"path": "finalization-diagnostic.json", "sha256": supervisor.file_digest(diagnostic)}],
        }

        class FakeFinalizerProcess:
            returncode = 2

            def __init__(self):
                self.polls = 0

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else self.returncode

            def communicate(self):
                supervisor.atomic_json(supervisor.EXECUTOR_ACK, terminal)
                return '{"status":"failed"}\n', ""

        def fake_finalizer(*_args, **_kwargs):
            return FakeFinalizerProcess()

        with patch.object(supervisor, "pid_alive", return_value=False), patch.object(
            supervisor, "executor_finalization_grace", return_value=(False, None)
        ), patch.object(supervisor.subprocess, "Popen", side_effect=fake_finalizer) as run:
            with patch.object(supervisor.time, "sleep", return_value=None):
                result = supervisor.run_executor_finalization(self.policy)
        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["ack_status"], "failed")
        self.assertIn("finalize_paper2_request.py", run.call_args.args[0][1])
        heartbeat = supervisor.load_json(supervisor.CONTROLLER_STATE)
        self.assertEqual(heartbeat["controller_status"], "finalizing")
        self.assertFalse(heartbeat["training_allowed"])

    def test_executor_finalizer_rejects_returncode_ack_mismatch(self):
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        dispatch = {
            "request_id": "mismatched-finalizer",
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "payload": {"pool_sha256": pool_sha},
            "strategy_evidence": [
                {"path": ".state/reference_resolution_budget_v2_plan.json"}
            ],
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, dispatch)
        checkpoint = self.root / "checkpoint.pkl"
        checkpoint.write_bytes(b"complete")
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, {
            "request_id": dispatch["request_id"],
            "attempt": 1,
            "status": "running",
            "worker_pid": 999999,
            "checkpoint_path": "checkpoint.pkl",
        })
        supervisor.atomic_json(
            self.root / ".state/reference_resolution_budget_v2.json",
            {"request": {"request_id": dispatch["request_id"], "attempt": 1}},
        )
        diagnostic = self.root / "finalization-diagnostic.json"
        diagnostic.write_text("{}\n", encoding="ascii")
        terminal = {
            "request_id": dispatch["request_id"],
            "attempt": 1,
            "status": "failed",
            "failure_class": "permanent",
            "checks": {"pool_sha256": pool_sha},
            "evidence": [
                {"path": diagnostic.name, "sha256": supervisor.file_digest(diagnostic)}
            ],
        }

        class MismatchedProcess:
            returncode = 0

            def poll(self):
                return self.returncode

            def communicate(self):
                supervisor.atomic_json(supervisor.EXECUTOR_ACK, terminal)
                return '{"status":"failed"}\n', ""

        with patch.object(supervisor, "pid_alive", return_value=False), patch.object(
            supervisor, "executor_finalization_grace", return_value=(False, None)
        ), patch.object(supervisor.subprocess, "Popen", return_value=MismatchedProcess()):
            with self.assertRaisesRegex(RuntimeError, "returncode does not match"):
                supervisor.run_executor_finalization(self.policy)

    def test_budget_v2_terminal_ack_requires_canonical_finalizer(self):
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        evidence = self.root / "executor-evidence.json"
        evidence.write_text("{}\n", encoding="ascii")
        dispatch = {
            "request_id": "racing-terminal-request",
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "strategy_revision": 2,
            "strategy_evidence": [
                {
                    "path": ".state/reference_resolution_budget_v2_plan.json",
                    "sha256": "A" * 64,
                }
            ],
            "payload": {"pool_sha256": pool_sha},
        }
        ack = {
            "request_id": dispatch["request_id"],
            "attempt": 1,
            "status": "failed",
            "failure_class": "scientific",
            "checks": {"pool_sha256": pool_sha},
            "evidence": [
                {"path": evidence.name, "sha256": supervisor.file_digest(evidence)}
            ],
        }
        valid, error = supervisor.validate_failed_ack(ack, pool_sha, dispatch)
        self.assertFalse(valid)
        self.assertIn("canonical paper2 finalizer", error)

    def test_supervisor_quarantines_racing_scientific_terminal_ack(self):
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        request_id = "racing-terminal-request"
        dispatch = {
            "request_id": request_id,
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "target_thread_id": self.policy["executor_thread_id"],
            "strategy_revision": 2,
            "strategy_evidence": [
                {
                    "path": ".state/reference_resolution_budget_v2_plan.json",
                    "sha256": "A" * 64,
                }
            ],
            "payload": {"pool_sha256": pool_sha},
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, dispatch)
        checkpoint = self.root / "checkpoint.pkl"
        checkpoint.write_bytes(b"complete")
        worker_evidence = self.root / ".state/reference_resolution_budget_v2.json"
        supervisor.atomic_json(
            worker_evidence,
            {"request": {"request_id": request_id, "attempt": 1}},
        )
        seal = {
            "schema_version": 1,
            "evidence_version": "paper2-executor-finalization-seal-v1",
            "request": {
                "request_id": request_id,
                "attempt": 1,
                "action": "joint_numerical_convergence",
            },
            "worker_pid": 999999,
        }
        supervisor.atomic_json(supervisor.executor_finalization_seal_path(dispatch), seal)
        bad_evidence = self.root / "bad-terminal-evidence.json"
        bad_evidence.write_text("{}\n", encoding="ascii")
        racing_ack = {
            "schema_version": 1,
            "thread_id": self.policy["executor_thread_id"],
            "request_id": request_id,
            "attempt": 1,
            "status": "failed",
            "failure_class": "scientific",
            "checkpoint_path": checkpoint.name,
            "checks": {"pool_sha256": pool_sha},
            "evidence": [
                {
                    "path": bad_evidence.name,
                    "sha256": supervisor.file_digest(bad_evidence),
                }
            ],
        }
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, racing_ack)
        canonical_evidence = self.root / "canonical-integrity-evidence.json"
        canonical_evidence.write_text("{}\n", encoding="ascii")
        canonical_ack = {
            "schema_version": 1,
            "finalizer_version": supervisor.PAPER2_FINALIZER_VERSION,
            "thread_id": self.policy["executor_thread_id"],
            "request_id": request_id,
            "attempt": 1,
            "status": "failed",
            "failure_class": "permanent",
            "checks": {
                "pool_sha256": pool_sha,
                "training_allowed": False,
                "finalizer_version": supervisor.PAPER2_FINALIZER_VERSION,
                "finalizer_verified_worker_dead": True,
                "finalization_classification": "execution_integrity_failure",
            },
            "evidence": [
                {
                    "path": canonical_evidence.name,
                    "sha256": supervisor.file_digest(canonical_evidence),
                }
            ],
        }

        class CanonicalFinalizerProcess:
            returncode = 2

            def poll(self):
                return self.returncode

            def communicate(self):
                active = supervisor.load_json(supervisor.EXECUTOR_ACK)
                self_test.assertEqual(active["status"], "running")
                self_test.assertTrue(active["checks"]["terminal_ack_rejected"])
                supervisor.atomic_json(supervisor.EXECUTOR_ACK, canonical_ack)
                return '{"status":"failed"}\n', ""

        self_test = self
        with patch.object(supervisor, "pid_alive", return_value=False), patch.object(
            supervisor, "executor_finalization_grace", return_value=(False, None)
        ), patch.object(
            supervisor.subprocess, "Popen", return_value=CanonicalFinalizerProcess()
        ):
            result = supervisor.run_executor_finalization(self.policy)
        self.assertEqual(result["status"], "finalized")
        final_ack = supervisor.load_json(supervisor.EXECUTOR_ACK)
        self.assertEqual(final_ack["failure_class"], "permanent")
        self.assertEqual(
            final_ack["checks"]["finalization_classification"],
            "execution_integrity_failure",
        )
        rejected = (
            self.root
            / ".state/finalization_diagnostics"
            / f"{request_id}-attempt1-rejected-ack.json"
        )
        self.assertEqual(supervisor.load_json(rejected), racing_ack)

    def test_audit_only_recovery_finalizes_without_worker_pid(self):
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        dispatch = {
            "request_id": "audit-only-request",
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "strategy_based_on": "source-request",
            "payload": {"pool_sha256": pool_sha},
            "strategy_evidence": [
                {"path": ".state/reference_resolution_budget_v2_plan.json"}
            ],
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, dispatch)
        checkpoint = self.root / "checkpoint.pkl"
        checkpoint.write_bytes(b"complete")
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, {
            "request_id": dispatch["request_id"],
            "attempt": 1,
            "status": "running",
            "worker_pid": None,
            "checkpoint_path": "checkpoint.pkl",
            "checks": {
                "audit_only_recovery": True,
                "finalization_ready": True,
            },
        })
        supervisor.atomic_json(
            self.root / ".state/reference_resolution_budget_v2.json",
            {"request": {"request_id": "source-request", "attempt": 1}},
        )
        diagnostic = self.root / "audit-only-diagnostic.json"
        diagnostic.write_text("{}\n", encoding="ascii")
        terminal = {
            "request_id": dispatch["request_id"],
            "attempt": 1,
            "status": "failed",
            "failure_class": "scientific",
            "checks": {"pool_sha256": pool_sha},
            "evidence": [
                {"path": diagnostic.name, "sha256": supervisor.file_digest(diagnostic)}
            ],
        }

        class AuditOnlyProcess:
            returncode = 2

            def poll(self):
                return self.returncode

            def communicate(self):
                supervisor.atomic_json(supervisor.EXECUTOR_ACK, terminal)
                return '{"status":"failed"}\n', ""

        with patch(
            "scripts.reference_budget_v2_lineage.validate_lineage", return_value={}
        ), patch.object(
            supervisor.subprocess, "Popen", return_value=AuditOnlyProcess()
        ):
            result = supervisor.run_executor_finalization(self.policy)
        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["ack_status"], "failed")

    def test_verified_replacement_activation_archives_failed_generation_and_advances(self):
        failed = {
            "schema_version": 1,
            "protocol_version": 2,
            "request_id": "abcdef123456",
            "target_thread_id": self.policy["executor_thread_id"],
            "stage": "paper2_pipeline",
            "action": "replacement_pool_generation",
            "status": "failed",
            "attempt": 1,
            "max_attempts": 3,
            "created_at": supervisor.now_iso(),
            "updated_at": supervisor.now_iso(),
            "ack_required": True,
            "terminal_failure": True,
            "failure_class": "scientific",
            "last_error": "generation completed; independent activation required",
            "payload": {"pool_sha256": "OLD-POOL"},
            "instruction": "generate only",
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, failed)
        audit = {
            "pool": {"sha256": "NEW-POOL"},
            "training_gates": {"replacement_pool_ready": True},
        }
        request = supervisor.update_dispatch("joint_numerical_convergence", self.policy, audit)
        self.assertEqual(request["action"], "joint_numerical_convergence")
        self.assertEqual(request["status"], "pending")
        self.assertEqual(request["payload"]["pool_sha256"], "NEW-POOL")
        history = supervisor.STATE / "dispatch_history" / "abcdef123456-attempt1.json"
        archived = supervisor.load_json(history)
        self.assertEqual(archived["request"]["status"], "failed")
        self.assertEqual(archived["request"]["failure_class"], "scientific")

    def test_unverified_replacement_failure_remains_terminal(self):
        failed = {
            "request_id": "abcdef654321",
            "action": "replacement_pool_generation",
            "status": "failed",
            "attempt": 1,
            "max_attempts": 3,
            "terminal_failure": True,
            "failure_class": "scientific",
            "payload": {"pool_sha256": "OLD-POOL"},
            "instruction": "generate only",
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, failed)
        audit = {
            "pool": {"sha256": "OLD-POOL"},
            "training_gates": {"replacement_pool_ready": False},
        }
        request = supervisor.update_dispatch("joint_numerical_convergence", self.policy, audit)
        self.assertEqual(request["request_id"], failed["request_id"])
        self.assertEqual(request["status"], "failed")

    def test_auto_transition_failure_is_fail_closed(self):
        completed = supervisor.subprocess.CompletedProcess(
            args=["python"], returncode=2, stdout="", stderr="evidence mismatch"
        )
        controller = {
            "dispatch": {
                "action": "joint_numerical_convergence",
                "status": "failed",
                "terminal_failure": True,
                "failure_class": "scientific",
            }
        }
        with patch.object(supervisor.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "evidence mismatch"):
                supervisor.run_auto_transition(controller)

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
                "thread_id": self.policy["executor_thread_id"],
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
            "outputs": [{"path": "legacy.pkl", "material": "legacy", "sha256": supervisor.file_digest(legacy)}],
            "paper_hashes": [{"path": "evidence.json", "md5": supervisor.file_digest(evidence, "md5")}],
        }
        valid, error = supervisor.validate_completed_ack(ack, "ABC", self.policy)
        self.assertFalse(valid)
        self.assertIn("immutable asset", error)

    def test_completed_ack_requires_exact_policy_paper_hash(self):
        output = self.root / "evidence.json"
        output.write_text("{}\n", encoding="ascii")
        paper = self.root / "paper.tex"
        paper.write_text("locked paper\n", encoding="ascii")
        locked_md5 = supervisor.file_digest(paper, "md5")
        self.policy["protected_files"] = [{"path": "paper.tex", "md5": locked_md5}]
        ack = {
            "checks": {"pool_sha256": "ABC"},
            "outputs": [
                {
                    "path": "evidence.json",
                    "material": "audit",
                    "sha256": supervisor.file_digest(output),
                }
            ],
            "paper_hashes": [{"path": "paper.tex", "md5": locked_md5}],
        }
        self.assertEqual(
            supervisor.validate_completed_ack(ack, "ABC", self.policy), (True, None)
        )

        paper.write_text("changed paper\n", encoding="ascii")
        changed_md5 = supervisor.file_digest(paper, "md5")
        ack["paper_hashes"] = [{"path": "paper.tex", "md5": changed_md5}]
        valid, error = supervisor.validate_completed_ack(ack, "ABC", self.policy)
        self.assertFalse(valid)
        self.assertIn("policy lock", error)

    def test_audited_gate_is_provisional_until_completed_ack(self):
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        request = {"request_id": "gate-request", "attempt": 1}
        evidence = self.root / "gate-evidence.json"
        supervisor.atomic_json(
            evidence,
            {
                "schema_version": 1,
                "evidence_version": "transactional-gate-v1",
                "passed": True,
                "pool_sha256": pool_sha,
                "request": request,
            },
        )
        supervisor.atomic_json(
            supervisor.GATE_STATE,
            {
                "schema_version": 1,
                "gates": {
                    "transactional_gate": {
                        "passed": True,
                        "evidence": [
                            {
                                "path": "gate-evidence.json",
                                "sha256": supervisor.file_digest(evidence),
                            }
                        ],
                    }
                },
            },
        )
        supervisor.atomic_json(
            supervisor.DISPATCH_REQUEST,
            {
                **request,
                "action": "transactional_action",
                "status": "in_progress",
            },
        )
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {**request, "status": "running"},
        )
        policy = copy.deepcopy(self.policy)
        policy["workflow"] = {
            "actions": [
                {
                    "action": "transactional_action",
                    "gate": "transactional_gate",
                    "evidence_version": "transactional-gate-v1",
                    "auditor": "python independent_auditor.py",
                }
            ],
            "required_before_training": ["transactional_gate"],
        }
        with patch.dict(
            supervisor.GATE_PAYLOAD_VERIFIERS,
            {"transactional_gate": lambda _payload, _pool: (True, None)},
        ):
            gates, _details = supervisor.verify_gate_evidence(
                policy, {"passed": True, "sha256": pool_sha}
            )
            self.assertFalse(gates["transactional_gate"])
            supervisor.atomic_json(
                supervisor.EXECUTOR_ACK,
                {**request, "status": "completed"},
            )
            gates, _details = supervisor.verify_gate_evidence(
                policy, {"passed": True, "sha256": pool_sha}
            )
            self.assertTrue(gates["transactional_gate"])

    def test_policy_integrity_lock_is_required_when_enabled(self):
        self.policy["integrity"]["enforce"] = True
        result = supervisor.verify_policy_integrity(self.policy)
        self.assertFalse(result["passed"])
        self.assertIn("lock", result["error"])

    def test_workflow_contract_rejects_unknown_ready_gate_but_allows_deferred_training(self):
        policy = {
            "workflow": {
                "actions": [
                    {"action": "unknown", "gate": "unknown_gate"},
                    {
                        "action": "future_training",
                        "gate": "future_training_gate",
                        "implementation_state": "deferred_until_pretraining_complete",
                        "requires_training_allowed": True,
                    },
                ],
                "required_before_training": [],
            }
        }
        with self.assertRaisesRegex(ValueError, "lacks an independent verifier"):
            supervisor.validate_workflow_contract(policy)
        policy["workflow"]["actions"] = [policy["workflow"]["actions"][1]]
        supervisor.validate_workflow_contract(policy)

    def test_joint_finalizer_profile_is_frozen_by_dispatch_evidence(self):
        policy = {
            "workflow": {
                "actions": [
                    {
                        "action": "joint_numerical_convergence",
                        "worker_evidence": ".state/joint_convergence_v2.json",
                        "finalizer": "scripts/finalize_audited_gate.py",
                    }
                ]
            }
        }
        normal = supervisor.action_finalization_spec(
            policy, "joint_numerical_convergence", {"strategy_evidence": []}
        )
        self.assertEqual(normal["worker_evidence"], ".state/joint_convergence_v2.json")
        diagnostic = supervisor.action_finalization_spec(
            policy,
            "joint_numerical_convergence",
            {
                "strategy_evidence": [
                    {
                        "path": ".state/reference_resolution_budget_v2_plan.json",
                        "sha256": "A" * 64,
                    }
                ]
            },
        )
        self.assertEqual(
            diagnostic["worker_evidence"],
            ".state/reference_resolution_budget_v2.json",
        )
        self.assertEqual(diagnostic["finalizer"], "scripts/finalize_paper2_request.py")

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
        numerical = pool_spec["expected_meta"]
        approved_candidate = {
            "requested_nG": int(numerical["nG"]),
            "Nxy": 256,
            "wavelength_step_nm": 5.0,
            "passed": True,
        }

        reference = supervisor.STATE / "reference_holdout_audit.json"
        supervisor.atomic_json(
            reference,
            {
                "schema_version": 1,
                "evidence_version": "paper2-reference-holdout-audit-v1",
                "protocol_revision": "v2_bound_holdout",
                "passed": True,
                "production_reference_approved": True,
                "classification": "reference_holdout_passed",
                "final_reference": {
                    "requested_nG": 450,
                    "Nxy": 768,
                    "wavelength_step_nm": 0.5,
                },
                "approved_protocol_candidate": approved_candidate,
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
                "protocol_revision": "v2_bound_holdout",
                "approved": True,
                "automatic_launch_authorized": True,
                "nG_requested": approved_candidate["requested_nG"],
                "Nxy": approved_candidate["Nxy"],
                "wavelength_step_nm": approved_candidate["wavelength_step_nm"],
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
                "evidence_version": "paper2-replacement-pool-audit-v1",
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

    def test_production_reference_requires_v2_revision_classification_and_final_reference(self):
        audit = {
            "evidence_version": "paper2-reference-holdout-audit-v1",
            "protocol_revision": "v2_bound_holdout",
            "classification": "reference_holdout_passed",
            "final_reference": copy.deepcopy(supervisor.REFERENCE_HOLDOUT_FINAL_REFERENCE),
            "passed": True,
            "production_reference_approved": True,
        }
        self.assertTrue(supervisor.production_reference_audit_approved(audit))
        for field in ("protocol_revision", "classification", "final_reference"):
            stale = copy.deepcopy(audit)
            stale.pop(field)
            self.assertFalse(supervisor.production_reference_audit_approved(stale))

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
                "thread_id": self.policy["executor_thread_id"],
                "status": "completed",
                "observed_at": supervisor.now_iso(),
                "outputs": [{"path": "pool_manifest.json", "material": "pool_manifest", "sha256": supervisor.file_digest(manifest)}],
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
                "thread_id": self.policy["executor_thread_id"],
                "status": "completed",
                "observed_at": supervisor.now_iso(),
                "outputs": [{"path": "output.json", "material": "evidence", "sha256": supervisor.file_digest(output)}],
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
        evidence, pool = self.write_d65_gate()
        gates, _ = supervisor.verify_gate_evidence(self.policy, pool)
        self.assertTrue(gates["d65_colorimetry"])
        evidence.write_text('{"tampered": true}\n', encoding="ascii")
        gates, _ = supervisor.verify_gate_evidence(self.policy, pool)
        self.assertFalse(gates["d65_colorimetry"])

    def test_gate_rejects_wrong_evidence_version(self):
        _, pool = self.write_d65_gate(evidence_version="paper2-d65-v0")
        gates, details = supervisor.verify_gate_evidence(
            self.policy, pool
        )
        self.assertFalse(gates["d65_colorimetry"])
        self.assertIn("evidence_version", details["d65_colorimetry"]["evidence"][0]["semantic_error"])

    def test_protocol_bound_gate_survives_active_pool_switch(self):
        self.write_d65_gate()
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

    def test_minimal_self_declared_gate_evidence_never_unlocks_training(self):
        gate_state = {"schema_version": 1, "gates": {}}
        for item in self.policy["workflow"]["actions"]:
            if item["gate"] == "pool_manifest_frozen":
                continue
            path = self.root / f"fake-{item['gate']}.json"
            supervisor.atomic_json(path, {
                "schema_version": 1,
                "passed": True,
                "evidence_version": item.get("evidence_version"),
                "pool_sha256": "ABC",
            })
            gate_state["gates"][item["gate"]] = {
                "passed": True,
                "evidence": [{"path": path.name, "sha256": supervisor.file_digest(path)}],
            }
        supervisor.atomic_json(supervisor.GATE_STATE, gate_state)
        gates, _ = supervisor.verify_gate_evidence(
            self.policy, {"passed": True, "sha256": "ABC", "records": 2}
        )
        self.assertFalse(gates["training_allowed"])
        self.assertTrue(all(
            gates[item["gate"]] is False
            for item in self.policy["workflow"]["actions"]
            if item["gate"] != "pool_manifest_frozen"
        ))

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
            if item.get("manual_only") is True:
                continue
            expected = item["action"]
            if item.get("requires_training_allowed"):
                gates["training_allowed"] = True
            self.assertEqual(supervisor.select_workflow_action(self.policy, gates), expected)
            gates[item["gate"]] = True
            if item["gate"] == "geometry_split_frozen":
                gates["training_allowed"] = True
        self.assertIsNone(supervisor.select_workflow_action(self.policy, gates))

    def test_preregistration_and_reference_precede_joint_while_full_pool_is_manual(self):
        gates = {item["gate"]: False for item in self.policy["workflow"]["actions"]}
        gates.update({
            "pool_complete": True,
            "strict_pool_validation": True,
            "pool_manifest_frozen": True,
            "d65_colorimetry": True,
        })
        self.assertEqual(
            supervisor.select_workflow_action(self.policy, gates),
            "multifidelity_preregistration",
        )
        gates["multifidelity_preregistered"] = True
        self.assertEqual(supervisor.select_workflow_action(self.policy, gates), "reference_resolution")
        gates["reference_resolution"] = True
        replacement = next(
            item
            for item in self.policy["workflow"]["actions"]
            if item["action"] == "replacement_pool_generation"
        )
        self.assertTrue(replacement["manual_only"])
        self.assertFalse(replacement["automatic_launch_authorized"])
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
            "paper2-joint-convergence-audit-v1",
        )
        self.assertIn(
            "run_joint_convergence_v2.py",
            actions["joint_numerical_convergence"]["runner"],
        )
        self.assertIn(
            "audit_joint_convergence_v2.py",
            actions["joint_numerical_convergence"]["auditor"],
        )
        self.assertEqual(
            actions["cross_solver_spectrum_validation"]["evidence_version"],
            "paper2-cross-solver-audit-v1",
        )
        self.assertIn(
            "run_cross_solver_validation_v2.py",
            actions["cross_solver_spectrum_validation"]["runner"],
        )
        self.assertIn(
            "audit_cross_solver_v2.py",
            actions["cross_solver_spectrum_validation"]["auditor"],
        )
        self.assertEqual(
            actions["replacement_pool_generation"]["evidence_version"],
            "paper2-replacement-pool-audit-v1",
        )
        self.assertIn(
            "audit_replacement_pool.py",
            actions["replacement_pool_generation"]["auditor"],
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
        self.assertIn("audit_joint_convergence_v2.py", joint_instruction)
        self.assertIn("activated replacement pool", joint_instruction)
        self.assertIn("launch_reference_resolution_holdout.py", reference_instruction)
        self.assertIn("run_cross_solver_validation_v2.py", cross_instruction)
        self.assertIn("audit_cross_solver_v2.py", cross_instruction)
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
                "thread_id": self.policy["executor_thread_id"],
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
            "strategy_revision": 2,
            "strategy_based_on": "failed-parent-request",
            "strategy_evidence": [
                {"path": "frozen-evidence.json", "sha256": "A" * 64}
            ],
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, active)
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, {
            "request_id": active["request_id"],
            "attempt": 1,
            "thread_id": self.policy["executor_thread_id"],
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
        self.assertEqual(result["strategy_revision"], active["strategy_revision"])
        self.assertEqual(result["strategy_based_on"], active["strategy_based_on"])
        self.assertEqual(result["strategy_evidence"], active["strategy_evidence"])

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
                "thread_id": self.policy["executor_thread_id"],
                "status": "running",
                "observed_at": supervisor.now_iso(),
                "lease_expires_at": lease,
            },
        )
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["status"], "pending")
        self.assertEqual(second["dispatch"]["attempt"], 2)

    def test_live_worker_with_expired_lease_blocks_concurrent_retry(self):
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        lease = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat(
            timespec="seconds"
        )
        supervisor.atomic_json(
            supervisor.EXECUTOR_ACK,
            {
                "request_id": request["request_id"],
                "attempt": request["attempt"],
                "thread_id": self.policy["executor_thread_id"],
                "status": "running",
                "observed_at": supervisor.now_iso(),
                "lease_expires_at": lease,
                "worker_pid": 4242,
            },
        )
        with patch.object(
            supervisor, "pid_alive", side_effect=lambda pid: int(pid) == 4242
        ):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"]["status"], "in_progress")
        self.assertEqual(second["dispatch"]["attempt"], 1)
        self.assertTrue(second["dispatch"]["recovery_blocked"])
        self.assertIn("concurrent recovery is blocked", second["dispatch"]["last_error"])

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
                "thread_id": self.policy["executor_thread_id"],
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
                "thread_id": self.policy["executor_thread_id"],
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
                "thread_id": self.policy["executor_thread_id"],
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
        self.assertEqual(advanced, failed)

        self.policy["strategy_override"].update({
            "decision": "transition_after_failure",
            "from_action": "joint_numerical_convergence",
        })
        advanced = supervisor.update_dispatch(
            "reference_resolution", self.policy, {"pool": {"sha256": pool_sha}}
        )
        self.assertNotEqual(advanced["request_id"], failed["request_id"])
        self.assertEqual(advanced["action"], "reference_resolution")
        self.assertEqual(advanced["status"], "pending")
        self.assertEqual(advanced["strategy_revision"], 2)
        self.assertEqual(advanced["strategy_based_on"], failed["request_id"])
        self.assertEqual(
            advanced["strategy_evidence"],
            self.policy["strategy_override"]["evidence"],
        )
        frozen = copy.deepcopy(advanced)
        repeated = supervisor.update_dispatch(
            "joint_numerical_convergence",
            self.policy,
            {"pool": {"sha256": "0" * 64}},
        )
        self.assertEqual(repeated, frozen)

    def test_terminal_joint_failure_relays_to_frozen_reference_request(self):
        pool_sha = supervisor.file_digest(self.root / "pool.pkl")
        failed = {
            "request_id": "failed-budget-v2-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "attempt": 1,
            "max_attempts": 3,
            "terminal_failure": True,
            "failure_class": "scientific",
            "strategy_revision": 2,
            "payload": {"pool": "pool.pkl", "pool_sha256": pool_sha},
            "instruction": "frozen failed request",
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, failed)
        evidence = self.root / "holdout-plan.json"
        evidence.write_text('{"plan_valid": true}\n', encoding="ascii")
        self.policy["strategy_override"] = {
            "enabled": True,
            "decision": "transition_after_failure",
            "revision": 3,
            "action": "reference_resolution",
            "from_action": "joint_numerical_convergence",
            "based_on_request_id": failed["request_id"],
            "instruction_append": "Launch only the hash-bound v2 holdout.",
            "evidence": [
                {"path": evidence.name, "sha256": supervisor.file_digest(evidence)}
            ],
        }
        self.write_status({"status": "running", "pid": 999999})
        with patch.object(supervisor, "pid_alive", return_value=False):
            first = supervisor.evaluate_once(self.policy)
        request = first["dispatch"]
        self.assertNotEqual(request["request_id"], failed["request_id"])
        self.assertEqual(request["action"], "reference_resolution")
        self.assertEqual(request["status"], "pending")
        self.assertEqual(request["strategy_revision"], 3)
        self.assertEqual(request["strategy_based_on"], failed["request_id"])
        self.assertEqual(first["controller_status"], "pending")
        self.assertEqual(first["pipeline_status"], "pending")
        self.assertFalse(first["pipeline_complete"])

        frozen = copy.deepcopy(request)
        with patch.object(supervisor, "pid_alive", return_value=False):
            second = supervisor.evaluate_once(self.policy)
        self.assertEqual(second["dispatch"], frozen)
        self.assertEqual(second["next_action"], "reference_resolution")
        self.assertFalse(second["pipeline_complete"])

    def test_strategy_revision_must_increase_and_request_id_binds_decision(self):
        repair = self.root / "repair_evidence.json"
        repair.write_text('{"passed": true}\n', encoding="ascii")
        failed = {
            "request_id": "failed-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "attempt": 1,
            "max_attempts": 3,
            "terminal_failure": True,
            "strategy_revision": 2,
        }
        self.policy["strategy_override"] = {
            "enabled": True,
            "decision": "retry_same_gate",
            "revision": 2,
            "action": failed["action"],
            "based_on_request_id": failed["request_id"],
            "instruction_append": "Use only the versioned repair.",
            "evidence": [
                {"path": repair.name, "sha256": supervisor.file_digest(repair)}
            ],
        }
        self.assertIsNone(supervisor.strategy_override(failed["action"], self.policy, failed))
        self.policy["strategy_override"]["revision"] = 3
        self.assertIsNotNone(supervisor.strategy_override(failed["action"], self.policy, failed))
        first = supervisor.make_dispatch_id(
            "paper2_pipeline", failed["action"], "ABC", 3,
            "retry_same_gate", "failed-request",
        )
        second = supervisor.make_dispatch_id(
            "paper2_pipeline", failed["action"], "ABC", 3,
            "transition_after_failure", "failed-request",
        )
        replay = supervisor.make_dispatch_id(
            "paper2_pipeline", failed["action"], "ABC", 3,
            "retry_same_gate", "another-request",
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, replay)

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
                "thread_id": self.policy["executor_thread_id"],
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
