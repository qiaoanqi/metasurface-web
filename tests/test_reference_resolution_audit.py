import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import audit_reference_resolution_result as audit
from scripts.audit_reference_resolution_result import (
    classify,
    physics_controls_pass,
    validate_result,
)


class ReferenceResolutionAuditTests(unittest.TestCase):
    def result_pair(self):
        wavelength = np.arange(380.0, 781.0, 1.0)
        expected = {
            "id": "task",
            "geometry_index": 0,
            "geometry": {"L": 200.0, "W": 150.0, "H": 300.0, "P": 400.0},
            "pol": "p",
            "requested_nG": 131,
            "retained_nG": 121,
            "Nxy": 256,
            "step_nm": 1,
            "wavelength_nm": wavelength,
        }
        result = {
            **expected,
            "status": "ok",
            "R": np.linspace(0.1, 0.9, wavelength.size),
            "time_s": 1.0,
        }
        result["T"] = 1.0 - result["R"]
        return result, expected

    def test_exact_result_validation(self):
        result, expected = self.result_pair()
        validate_result(result, expected)
        result["Nxy"] = 384
        with self.assertRaisesRegex(ValueError, "Nxy"):
            validate_result(result, expected)

    def test_versioned_result_schemas_are_strict(self):
        result, expected = self.result_pair()
        historical = dict(result)
        historical.pop("time_s")
        validate_result(
            historical,
            expected,
            schema=audit.JOINT_V1_1_RESULT_SCHEMA,
        )

        with self.assertRaisesRegex(ValueError, "field set mismatch"):
            validate_result(historical, expected)

        historical["unknown"] = "rejected"
        with self.assertRaisesRegex(ValueError, "field set mismatch"):
            validate_result(
                historical,
                expected,
                schema=audit.JOINT_V1_1_RESULT_SCHEMA,
            )

    def test_physics_controls_are_all_required(self):
        payload = {
            "solver_verdict": "pass",
            "independent_checks": {
                "empty_layer_fresnel": {"R": 0.04, "analytic_R": 0.04, "rt": 1.0},
                "rotation_max_dR": 1e-12,
                "rotation_max_dT": 1e-12,
                "lw_circle_bitwise": True,
            },
        }
        self.assertTrue(physics_controls_pass(payload))
        payload["independent_checks"]["lw_circle_bitwise"] = False
        self.assertFalse(physics_controls_pass(payload))

    def test_classification_keeps_reference_and_production_separate(self):
        self.assertEqual(classify(False, True, False), "reference_requires_followup")
        self.assertEqual(classify(True, False, False), "implementation_control_failure")
        self.assertEqual(classify(True, True, False), "historical_production_budget_rejected")
        self.assertEqual(classify(True, True, True), "historical_5nm_sampling_rejected")

    def test_only_execution_failure_can_be_replaced(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT) as temporary:
            output = Path(temporary) / "audit.json"
            execution_failure = {
                "passed": False,
                "classification": "execution_integrity_failure",
            }
            scientific_result = {
                "passed": False,
                "classification": "reference_spatial_budget_insufficient_order",
            }
            audit.persist_audit(output, execution_failure)
            audit.persist_audit(output, scientific_result)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), scientific_result
            )

            with self.assertRaisesRegex(ValueError, "bump the evidence version"):
                audit.persist_audit(
                    output,
                    {"passed": True, "classification": "unexpected_rewrite"},
                )


class ReferenceResolutionEndToEndAuditTests(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def write_pickle(path: Path, payload: dict) -> None:
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def record(expected: dict, reflectance: float = 0.2) -> dict:
        wavelength = np.asarray(expected["wavelength_nm"], dtype=float)
        reflection = np.full(wavelength.shape, reflectance, dtype=float)
        return {
            **expected,
            "wavelength_nm": wavelength,
            "status": "ok",
            "R": reflection,
            "T": 1.0 - reflection,
            "time_s": 1.0,
        }

    def fixture(self, failed_axis: str | None = None, worker_passed: bool | None = None):
        temporary = tempfile.TemporaryDirectory(dir=audit.ROOT)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        paths = {
            "source": root / "joint_convergence_v1_1.json",
            "v1_checkpoint": root / "joint_convergence_v1_1_checkpoint.pkl",
            "checkpoint": root / "reference_resolution_v1_checkpoint.pkl",
            "evidence": root / "reference_resolution_v1.json",
            "physics": root / "physics_audit.json",
            "plan": root / "reference_resolution_v1_plan.json",
        }
        selected = [
            {
                "L": 140.0 + index,
                "W": 100.0 + index,
                "H": 300.0 + index,
                "P": 400.0 + index,
                "selection": "test_frozen_case",
            }
            for index in range(8)
        ]
        self.write_json(
            paths["source"],
            {
                "schema_version": 1,
                "evidence_version": "paper2-joint-convergence-v1.1",
                "passed": False,
            },
        )
        production_results = {}
        production_wavelength = np.arange(380.0, 781.0, 1.0)
        for index, geometry in enumerate(selected):
            for pol in audit.POLS:
                identifier = audit.v1_1_task_id(index, pol, audit.PRODUCTION)
                expected = {
                    "id": identifier,
                    "geometry_index": index,
                    "geometry": geometry,
                    "pol": pol,
                    "requested_nG": audit.PRODUCTION[0],
                    "retained_nG": audit.retained_order(
                        audit.PRODUCTION[0], geometry["P"]
                    ),
                    "Nxy": audit.PRODUCTION[1],
                    "step_nm": 1,
                    "wavelength_nm": production_wavelength,
                }
                production_results[identifier] = self.record(expected)
                production_results[identifier].pop("time_s")
        v1_checkpoint = {
            "meta": {"selected_geometries": selected},
            "results": production_results,
        }
        self.write_pickle(paths["v1_checkpoint"], v1_checkpoint)

        runtime_hashes = {
            name: audit.file_digest(audit.ROOT / name)
            for name in audit.EXPECTED_RUNTIME_PATHS
        }
        baseline = {
            "path": audit.relative_path(paths["v1_checkpoint"]),
            "sha256": audit.file_digest(paths["v1_checkpoint"]),
        }
        source = {
            "path": audit.relative_path(paths["source"]),
            "sha256": audit.file_digest(paths["source"]),
        }
        meta = {
            "version": audit.REFERENCE_VERSION,
            "pool_sha256": "A" * 64,
            "selected_geometries": selected,
            "selection_source": source,
            "baseline_checkpoint": baseline,
            "expected_tasks": audit.EXPECTED_TASKS,
            "configs_1nm": [list(config) for config in audit.CONFIGS_1NM],
            "fine_config": list(audit.FINE_CONFIG),
            "fine_step_nm": 0.5,
            "thresholds": audit.EXPECTED_THRESHOLDS,
            "runtime_hashes": runtime_hashes,
        }
        results = {
            task["id"]: self.record(task) for task in audit.build_tasks(selected)
        }
        if failed_axis is not None:
            first_config, first_step, _second_config, _second_step = audit.AXIS_SPECS[
                failed_axis
            ]
            if failed_axis == "spectral":
                target_config, target_step = audit.FINE_CONFIG, 0.5
            else:
                target_config, target_step = first_config, first_step
            identifier = audit.reference_task_id(
                0, "p", target_config, target_step
            )
            results[identifier] = self.record(results[identifier], reflectance=0.8)
        checkpoint = {"meta": meta, "results": results}
        self.write_pickle(paths["checkpoint"], checkpoint)
        plan = audit.build_reference_plan(meta)
        self.write_json(paths["plan"], plan)
        raw_passed = failed_axis is None
        evidence = {
            "schema_version": 1,
            "evidence_version": audit.REFERENCE_VERSION,
            "passed": raw_passed if worker_passed is None else worker_passed,
            "pool_sha256": meta["pool_sha256"],
            "thresholds": audit.EXPECTED_THRESHOLDS,
            "selection": selected,
            "checkpoint": {
                "path": audit.relative_path(paths["checkpoint"]),
                "sha256": audit.file_digest(paths["checkpoint"]),
                "tasks": audit.EXPECTED_TASKS,
            },
            "input_evidence": source,
            "baseline_checkpoint": baseline,
            "runtime_hashes": runtime_hashes,
        }
        self.write_json(paths["evidence"], evidence)
        self.write_json(
            paths["physics"],
            {
                "solver_verdict": "pass",
                "independent_checks": {
                    "empty_layer_fresnel": {
                        "R": 0.04,
                        "analytic_R": 0.04,
                        "rt": 1.0,
                    },
                    "rotation_max_dR": 1e-12,
                    "rotation_max_dT": 1e-12,
                    "lw_circle_bitwise": True,
                },
            },
        )
        return paths, audit.file_digest(paths["plan"])

    def build(self, paths: dict, plan_sha256: str) -> dict:
        return audit.build_audit(
            paths["evidence"],
            paths["checkpoint"],
            paths["v1_checkpoint"],
            paths["physics"],
            paths["plan"],
            expected_plan_sha256=plan_sha256,
        )

    def rebind_checkpoint(self, paths: dict) -> None:
        evidence = json.loads(paths["evidence"].read_text(encoding="utf-8"))
        evidence["checkpoint"]["sha256"] = audit.file_digest(paths["checkpoint"])
        self.write_json(paths["evidence"], evidence)

    def test_raw_checkpoint_passes_without_trusting_worker_comparisons(self):
        paths, plan_sha256 = self.fixture()
        result = self.build(paths, plan_sha256)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failure_axes"], [])
        self.assertTrue(result["worker_claim"]["matches_independent_recomputation"])

    def test_tampered_worker_passed_cannot_override_raw_axis_failure(self):
        paths, plan_sha256 = self.fixture(failed_axis="order", worker_passed=True)
        result = self.build(paths, plan_sha256)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_axes"], ["order"])
        self.assertFalse(result["worker_claim"]["matches_independent_recomputation"])
        self.assertEqual(result["classification"], "worker_evidence_integrity_failure")

    def test_plan_runtime_and_meta_tampering_are_rejected(self):
        paths, plan_sha256 = self.fixture()
        plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
        plan["decision_rule"] = "tampered"
        self.write_json(paths["plan"], plan)
        with self.assertRaisesRegex(ValueError, "plan SHA256"):
            self.build(paths, plan_sha256)

        paths, _plan_sha256 = self.fixture()
        with paths["checkpoint"].open("rb") as handle:
            checkpoint = pickle.load(handle)
        checkpoint["meta"]["runtime_hashes"]["rcwa_batch.py"] = "0" * 64
        self.write_pickle(paths["checkpoint"], checkpoint)
        self.rebind_checkpoint(paths)
        evidence = json.loads(paths["evidence"].read_text(encoding="utf-8"))
        evidence["runtime_hashes"] = checkpoint["meta"]["runtime_hashes"]
        self.write_json(paths["evidence"], evidence)
        self.write_json(paths["plan"], audit.build_reference_plan(checkpoint["meta"]))
        with self.assertRaisesRegex(ValueError, "runtime hash mismatch"):
            self.build(paths, audit.file_digest(paths["plan"]))

        paths, plan_sha256 = self.fixture()
        with paths["checkpoint"].open("rb") as handle:
            checkpoint = pickle.load(handle)
        checkpoint["meta"]["fine_step_nm"] = 1.0
        self.write_pickle(paths["checkpoint"], checkpoint)
        self.rebind_checkpoint(paths)
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.build(paths, plan_sha256)

    def test_each_single_axis_over_threshold_is_rejected(self):
        expected_classifications = {
            "order": "reference_spatial_budget_insufficient_order",
            "grid": "reference_spatial_budget_insufficient_grid",
            "spectral": "reference_spectral_resolution_insufficient",
        }
        for axis, classification in expected_classifications.items():
            with self.subTest(axis=axis):
                paths, plan_sha256 = self.fixture(failed_axis=axis)
                result = self.build(paths, plan_sha256)
                self.assertFalse(result["passed"])
                self.assertEqual(result["failure_axes"], [axis])
                self.assertEqual(result["classification"], classification)
                self.assertFalse(result["reference_axes"][axis]["all_lt_2_3"])

    def test_wavelength_point_count_is_strict(self):
        paths, plan_sha256 = self.fixture()
        with paths["checkpoint"].open("rb") as handle:
            checkpoint = pickle.load(handle)
        identifier = audit.reference_task_id(0, "p", audit.FINE_CONFIG, 0.5)
        for field in ("wavelength_nm", "R", "T"):
            checkpoint["results"][identifier][field] = checkpoint["results"][identifier][
                field
            ][:-1]
        self.write_pickle(paths["checkpoint"], checkpoint)
        self.rebind_checkpoint(paths)
        with self.assertRaisesRegex(ValueError, "wavelength mismatch"):
            self.build(paths, plan_sha256)


if __name__ == "__main__":
    unittest.main()
