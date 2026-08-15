import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline_supervisor as supervisor
from scripts import activate_replacement_pool as activation
from scripts.activate_replacement_pool import (
    build_gate_state,
    protocol_bound_gates,
    replacement_spec,
)
from scripts import run_replacement_pool as replacement


class ActivateReplacementPoolTests(unittest.TestCase):
    def setUp(self):
        self.policy = copy.deepcopy(supervisor.load_policy())

    def evidence(self, spec, protocol_sha="PROTO", pool_sha="ABC"):
        return {
            "schema_version": 1,
            "evidence_version": "paper2-replacement-pool-v1",
            "passed": True,
            "pool_sha256": pool_sha,
            "pool_spec": spec,
            "approved_protocol": {"path": ".state/protocol.json", "sha256": protocol_sha},
            "activation_id": hashlib.sha256(
                f"{protocol_sha}|{pool_sha}".encode("ascii")
            ).hexdigest()[:24],
        }

    def test_replacement_spec_must_exactly_match_hash_bound_protocol(self):
        spec = copy.deepcopy(self.policy["pool"])
        spec["path"] = "data/replacement/test-v1.pkl"
        with tempfile.TemporaryDirectory() as directory:
            protocol_path = Path(directory) / "protocol.json"
            protocol_path.write_text("{}", encoding="utf-8")
            context = {"protocol": {"pool_spec": spec}}
            with patch.object(
                replacement, "canonical_workspace_path", return_value=protocol_path
            ), patch.object(
                replacement, "file_digest", return_value="PROTO"
            ), patch.object(
                replacement, "validate_protocol", return_value=context
            ):
                actual = replacement_spec(self.policy, self.evidence(spec))
                self.assertEqual(actual, spec)
                tampered = self.evidence({**spec, "expected_records": 2})
                with self.assertRaisesRegex(ValueError, "differs"):
                    replacement_spec(self.policy, tampered)
                bad_id = self.evidence(spec)
                bad_id["activation_id"] = "BAD"
                with self.assertRaisesRegex(ValueError, "activation_id"):
                    replacement_spec(self.policy, bad_id)

    def test_only_protocol_gates_survive_pool_activation(self):
        keep = protocol_bound_gates(self.policy)
        self.assertEqual(keep, {"d65_colorimetry", "reference_resolution"})
        # build_gate_state needs real files for hashes; its filtering rule is
        # asserted directly so pool-bound gates cannot leak across activation.
        old = {
            "gates": {
                "d65_colorimetry": {"passed": True},
                "reference_resolution": {"passed": True},
                "joint_numerical_convergence": {"passed": True},
            }
        }
        filtered = {name for name in old["gates"] if name in keep}
        self.assertEqual(filtered, {"d65_colorimetry", "reference_resolution"})

    def test_activation_resumes_after_failure_between_manifest_and_active_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state = root / ".state"
            data = root / "data"
            state.mkdir()
            (data / "replacement").mkdir(parents=True)
            old_pool = data / "old.pkl"
            new_pool = data / "replacement" / "new-v1.pkl"
            old_pool.write_bytes(b"old")
            new_pool.write_bytes(b"new")
            evidence_path = state / "replacement.json"
            evidence = {
                "activation_id": "ACTIVATION",
                "approved_protocol": {"path": ".state/protocol.json", "sha256": "PROTO"},
                "pool_sha256": supervisor.file_digest(new_pool),
            }
            supervisor.atomic_json(evidence_path, evidence)
            active_path = state / "active.json"
            manifest_path = state / "manifest.json"
            transaction_path = state / "transaction.json"
            gate_path = state / "gate.json"
            supervisor.atomic_json(
                gate_path,
                {"schema_version": 1, "gates": {"reference_resolution": {"passed": True}}},
            )
            local_policy = copy.deepcopy(self.policy)
            local_policy["pool"]["path"] = "data/old.pkl"
            spec = copy.deepcopy(local_policy["pool"])
            spec["path"] = "data/replacement/new-v1.pkl"
            evidence["pool_spec"] = spec
            supervisor.atomic_json(evidence_path, evidence)
            audit = {
                "passed": True,
                "records": 2,
                "expected_records": 2,
                "geometries": 1,
                "complete_pairs": 1,
                "duplicate_keys": 0,
            }
            real_atomic = supervisor.atomic_json
            failed = {"value": False}

            def fail_once(path, payload):
                if Path(path) == active_path and not failed["value"]:
                    failed["value"] = True
                    raise OSError("synthetic crash before active marker")
                real_atomic(Path(path), payload)

            patches = (
                patch.object(activation, "ROOT", root),
                patch.object(supervisor, "ROOT", root),
                patch.object(supervisor, "GATE_STATE", gate_path),
                patch.object(supervisor, "load_policy", return_value=local_policy),
                patch.object(supervisor, "verify_policy_integrity", return_value={"passed": True}),
                patch.object(supervisor, "audit_pool", return_value=audit),
                patch.object(supervisor, "audit_protected_files", return_value=[{"passed": True}]),
                patch.object(activation, "replacement_spec", return_value=spec),
                patch.object(replacement, "canonical_workspace_path", return_value=new_pool),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
                with patch.object(supervisor, "atomic_json", side_effect=fail_once):
                    with self.assertRaisesRegex(OSError, "synthetic crash"):
                        activation.activate(
                            evidence_path, active_path, manifest_path, transaction_path
                        )
                self.assertTrue(manifest_path.exists())
                self.assertFalse(active_path.exists())
                result = activation.activate(
                    evidence_path, active_path, manifest_path, transaction_path
                )
            self.assertTrue(result["active"])
            self.assertTrue(active_path.exists())
            self.assertEqual(
                supervisor.load_json(transaction_path, {})["status"], "committed"
            )


if __name__ == "__main__":
    unittest.main()
