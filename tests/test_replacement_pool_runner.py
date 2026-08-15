import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_replacement_pool as runner


def policy(root: Path) -> dict:
    return {
        "protected_files": [{"path": "paper_oe.tex", "md5": "X"}],
        "immutable_assets": [{"path": "data/rcwa_5k.pkl", "md5": "Y"}],
        "pool": {
            "path": "data/old.pkl",
            "range_tolerance": 1e-8,
            "pointwise_conservation_tolerance": 1e-6,
            "stored_value_tolerance": 1e-9,
            "quality_tolerance": 0.05,
        },
    }


def protocol(root: Path, samples: int = 2) -> dict:
    wavelength = runner.colorimetry.wavelength_grid(5.0)
    expected_meta = {
        "seed": 2026,
        "nG": 131,
        "Nxy": 256,
        "material": "TiO2",
        "substrate": "SiO2",
        "background": "air",
        "pols": ["p", "s"],
        "n_samples": samples,
        "sampler_version": "test-sampler",
        "quality_rule": "lossless |R_plus_T_mean - 1.0| <= 0.05",
        "wavelength_step_nm": 5.0,
        "colorimetry_version": runner.colorimetry.COLORIMETRY_VERSION,
        "axis_canonicalization": "L=max(raw_axes), W=min(raw_axes); recompute p/s on canonical axes",
    }
    return {
        "schema_version": 1,
        "evidence_version": runner.PROTOCOL_VERSION,
        "approved": True,
        "samples": samples,
        "seed": 2026,
        "nG_requested": 131,
        "nG_retained": 121,
        "Nxy": 256,
        "material": "TiO2",
        "substrate": "SiO2",
        "background": "air",
        "sampler_version": "test-sampler",
        "quality_rule": "lossless |R_plus_T_mean - 1.0| <= 0.05",
        "wavelength_step_nm": 5.0,
        "max_same_config_attempts": 2,
        "source_reference_gate": {"path": ".state/reference.json", "sha256": "A"},
        "pool_spec": {
            "path": "data/replacement/test-v1.pkl",
            "expected_records": samples * 2,
            "wavelength_nm": wavelength.tolist(),
            "required_record_fields": sorted(
                runner.BASE_REQUIRED_FIELDS | runner.REPLACEMENT_REQUIRED_FIELDS
            ),
            "polarizations": ["p", "s"],
            "material": "TiO2",
            "substrate": "SiO2",
            "nG_requested": 131,
            "lossless": True,
            "range_tolerance": 1e-8,
            "pointwise_conservation_tolerance": 1e-6,
            "stored_value_tolerance": 1e-9,
            "quality_tolerance": 0.05,
            "expected_meta": expected_meta,
            "resume_command": "python scripts/run_replacement_pool.py --approved-protocol .state/protocol.json --resume",
        },
    }


class ReplacementRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        (self.root / "data" / "replacement").mkdir(parents=True)
        (self.root / "data" / "old.pkl").write_bytes(b"old")
        (self.root / "data" / "rcwa_5k.pkl").write_bytes(b"legacy")
        (self.root / "paper_oe.tex").write_text("paper", encoding="utf-8")
        self.root_patch = patch.object(runner, "ROOT", self.root)
        self.output_patch = patch.object(
            runner, "ALLOWED_OUTPUT_DIRECTORY", (self.root / "data" / "replacement").resolve()
        )
        self.root_patch.start()
        self.output_patch.start()

    def tearDown(self):
        self.output_patch.stop()
        self.root_patch.stop()
        self.directory.cleanup()

    def test_pool_spec_is_exact_and_cannot_weaken_tolerances(self):
        approved = protocol(self.root)
        output, wavelength = runner.validate_pool_spec(
            approved["pool_spec"], approved, policy(self.root)
        )
        self.assertEqual(output, self.root / "data" / "replacement" / "test-v1.pkl")
        self.assertEqual(wavelength.size, 81)
        approved["pool_spec"]["pointwise_conservation_tolerance"] = 0.1
        with self.assertRaisesRegex(ValueError, "weakens"):
            runner.validate_pool_spec(approved["pool_spec"], approved, policy(self.root))

    def test_path_alias_cannot_target_old_or_protected_pool(self):
        approved = protocol(self.root)
        approved["pool_spec"]["path"] = "data/replacement/../old.pkl"
        with self.assertRaises(ValueError):
            runner.validate_pool_spec(approved["pool_spec"], approved, policy(self.root))
        approved = protocol(self.root)
        approved["pool_spec"]["path"] = "data/replacement/../../paper_oe.tex"
        with self.assertRaises(ValueError):
            runner.validate_pool_spec(approved["pool_spec"], approved, policy(self.root))

    def test_task_matrix_is_deterministic_and_paired(self):
        approved = protocol(self.root)
        context = {
            "protocol": approved,
            "params": [(120.0, 100.0, 300.0, 400.0), (160.0, 90.0, 240.0, 420.0)],
            "wavelength": runner.colorimetry.wavelength_grid(5.0),
        }
        first = runner.build_tasks(context)
        second = runner.build_tasks(context)
        self.assertEqual(runner.task_manifest_hash(first), runner.task_manifest_hash(second))
        self.assertEqual([task["pol"] for task in first], ["p", "s", "p", "s"])
        self.assertEqual(len({task["id"] for task in first}), 4)

    def test_axis_canonicalization_is_deterministic(self):
        params = [(90.0, 140.0, 300.0, 400.0), (180.0, 100.0, 250.0, 420.0)]
        canonical = runner.canonicalize_params(params)
        self.assertEqual(canonical[0], (140.0, 90.0, 300.0, 400.0))
        self.assertEqual(canonical[1], (180.0, 100.0, 250.0, 420.0))
        self.assertEqual(
            runner.geometry_manifest_hash(canonical),
            runner.geometry_manifest_hash(runner.canonicalize_params(params)),
        )

    def test_retry_never_changes_scientific_configuration(self):
        approved = protocol(self.root, samples=1)
        context = {
            "protocol": approved,
            "params": [(120.0, 100.0, 300.0, 400.0)],
            "wavelength": runner.colorimetry.wavelength_grid(5.0),
        }
        task = runner.build_tasks(context)[0]
        R = np.linspace(0.1, 0.9, 81)
        T = 1.0 - R
        with patch.object(
            runner,
            "rcwa_spectrum",
            side_effect=[RuntimeError("transient"), (R, T)],
        ) as solve:
            result = runner.run_task(task)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["record"]["attempts"], 2)
        self.assertEqual(solve.call_count, 2)
        for call in solve.call_args_list:
            self.assertEqual(call.kwargs["nG_req"], 131)
            self.assertEqual(call.kwargs["Nxy"], 256)
            self.assertEqual(call.kwargs["background"], "air")

    def test_sqlite_checkpoint_resume_preserves_identity_and_results(self):
        checkpoint_path = self.root / ".state" / "replacement.sqlite"
        identity = {"runner_version": "v1", "protocol_sha256": "A", "task_manifest_sha256": "B"}
        checkpoint = runner.Checkpoint(checkpoint_path, identity, resume=False)
        checkpoint.store({"id": "g-p", "ordinal": 0, "status": "ok", "record": {"x": 1}})
        checkpoint.close()
        resumed = runner.Checkpoint(checkpoint_path, identity, resume=True)
        self.assertEqual(resumed.successful_ids(), {"g-p"})
        self.assertEqual(resumed.results()[0]["record"], {"x": 1})
        resumed.close()
        with self.assertRaises(ValueError):
            runner.Checkpoint(
                checkpoint_path,
                {**identity, "protocol_sha256": "tampered"},
                resume=True,
            )

    def test_single_instance_lock_rejects_live_owner(self):
        lock_path = self.root / ".state" / "replacement.lock"
        with runner.RunLock(lock_path, "A"):
            with patch.object(runner, "pid_alive", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    runner.RunLock(lock_path, "A").__enter__()
        self.assertFalse(lock_path.exists())

    def test_preflight_never_creates_output(self):
        approved = protocol(self.root, samples=1)
        params = [(120.0, 100.0, 300.0, 400.0)]
        approved["geometry_manifest_sha256"] = runner.geometry_manifest_hash(params)
        protocol_path = self.root / ".state" / "protocol.json"
        protocol_path.parent.mkdir()
        protocol_path.write_text(json.dumps(approved), encoding="utf-8")
        fake_context = {
            "protocol": approved,
            "protocol_path": protocol_path,
            "protocol_sha256": runner.file_digest(protocol_path),
            "policy": policy(self.root),
            "output": self.root / approved["pool_spec"]["path"],
            "wavelength": runner.colorimetry.wavelength_grid(5.0),
            "params": params,
        }
        args = argparse.Namespace(
            approved_protocol=runner.relative_path(protocol_path),
            checkpoint=".state/checkpoint.sqlite",
            evidence=".state/evidence.json",
            lock=".state/lock",
            n_jobs=1,
            progress_every=1,
            resume=False,
            preflight=True,
        )
        with patch.object(runner, "validate_protocol", return_value=fake_context):
            result = runner.run(args)
        self.assertTrue(result["preflight"])
        self.assertFalse(fake_context["output"].exists())


if __name__ == "__main__":
    unittest.main()
