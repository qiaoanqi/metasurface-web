import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

import paper2_colorimetry_fine as colorimetry
from scripts import run_joint_convergence_v2 as joint
from scripts import run_reference_resolution_escalation as base
from scripts import run_reference_resolution_holdout as holdout


class JointConvergenceV2Tests(unittest.TestCase):
    def build_fixture(self, directory: Path):
        production_wavelength = colorimetry.wavelength_grid(5.0)
        reference_wavelength = colorimetry.wavelength_grid(0.5)
        production_R = np.full(production_wavelength.size, 0.3)
        production_T = 1.0 - production_R
        reference_R = np.full(reference_wavelength.size, 0.3)
        reference_T = 1.0 - reference_R
        labels = colorimetry.spectrum_to_labels_d65(production_R, production_wavelength)
        cases = []
        records = []
        reference_results = {}
        for index in range(32):
            long_axis = 140.0 + index
            short_axis = 90.0 + index * 0.25
            swapped = index % 2 == 1
            L, W = (short_axis, long_axis) if swapped else (long_axis, short_axis)
            geometry = {"L": L, "W": W, "H": 250.0 + index, "P": 420.0 + index}
            cases.append(geometry)
            canonical = (long_axis, short_axis, geometry["H"], geometry["P"])
            for pol in base.POLS:
                records.append(
                    {
                        "geometry_id": f"g{index}",
                        "L": canonical[0],
                        "W": canonical[1],
                        "H": canonical[2],
                        "P": canonical[3],
                        "pol": pol,
                        "wl_nm": production_wavelength,
                        "R": production_R,
                        "T": production_T,
                        "xyz": labels["xyz"],
                        "lab": labels["lab"],
                        "srgb_display": labels["srgb_display"],
                        "label_provenance_version": colorimetry.COLORIMETRY_VERSION,
                        "nG_requested": 290,
                        "nG_retained": 289,
                        "Nxy": 384,
                        "wavelength_step_nm": 5.0,
                    }
                )
                source_pol = {"p": "s", "s": "p"}[pol] if swapped else pol
                identifier = holdout.result_id(index, source_pol, base.FINE_CONFIG, 0.5)
                reference_results[identifier] = {
                    "status": "ok",
                    "wavelength_nm": reference_wavelength,
                    "R": reference_R,
                    "T": reference_T,
                }
        pool_path = directory / "pool.pkl"
        with pool_path.open("wb") as handle:
            pickle.dump({"records": records}, handle)
        return (
            {"pool_path": pool_path},
            {"plan": {"combined_cases": cases}, "results": reference_results},
            records,
        )

    def test_exact_canonical_pool_passes_and_swaps_reference_polarizations(self):
        with tempfile.TemporaryDirectory() as directory:
            context, reference, _records = self.build_fixture(Path(directory))
            result = joint.evaluate(context, reference)
        self.assertTrue(result["passed"])
        self.assertEqual(result["joint_dE00"]["count"], 32)
        self.assertEqual(
            sum(row["axes_swapped_from_reference"] for row in result["rows"]), 32
        )

    def test_tampered_derived_label_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            context, reference, records = self.build_fixture(Path(directory))
            records[0]["lab"] = np.asarray(records[0]["lab"]) + [1.0, 0.0, 0.0]
            with context["pool_path"].open("wb") as handle:
                pickle.dump({"records": records}, handle)
            result = joint.evaluate(context, reference)
        self.assertFalse(result["passed"])
        self.assertEqual(result["label_failures"], [{"geometry_index": 0, "pol": "p"}])

    def test_missing_polarization_fails_exact_pair_check(self):
        with tempfile.TemporaryDirectory() as directory:
            context, reference, records = self.build_fixture(Path(directory))
            records.pop()
            with context["pool_path"].open("wb") as handle:
                pickle.dump({"records": records}, handle)
            result = joint.evaluate(context, reference)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["exact_32_complete_p_s_geometries"])


if __name__ == "__main__":
    unittest.main()
