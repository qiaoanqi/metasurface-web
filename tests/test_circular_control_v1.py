import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import run_circular_control_v1 as circular
from scripts import run_replacement_pool as replacement


class CircularControlTests(unittest.TestCase):
    def test_area_equivalent_selection_is_deterministic_and_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "pool.pkl"
            records = []
            for index in range(24):
                geometry = (220.0 + index, 100.0, 180.0 + index, 500.0 + index)
                identifier = replacement.geometry_id(geometry)
                for pol in ("p", "s"):
                    records.append(
                        {
                            "geometry_id": identifier,
                            "L": geometry[0],
                            "W": geometry[1],
                            "H": geometry[2],
                            "P": geometry[3],
                            "pol": pol,
                        }
                    )
            with pool.open("wb") as handle:
                pickle.dump({"records": records}, handle)
            selected = circular.load_source_geometries(pool)
            self.assertEqual(len(selected), 12)
            self.assertEqual(len({item["control_id"] for item in selected}), 12)
            for item in selected:
                self.assertGreater(item["D"], 0.0)
                self.assertLess(item["D"], item["P"])

    def test_equal_polarizations_pass_and_raw_difference_is_measured(self):
        wavelength = np.arange(380.0, 785.0, 5.0)
        reflection = np.linspace(0.1, 0.4, wavelength.size)
        transmission = 1.0 - reflection
        meta = {
            "wavelength_nm": wavelength,
            "thresholds": {
                "pointwise_conservation_lte": 1e-6,
                "polarization_spectrum_max_abs_lte": 1e-7,
                "polarization_dE00_lte": 0.01,
            },
        }
        result = {
            "id": "circle",
            "status": "ok",
            "spectra": {
                "p": {"R": reflection, "T": transmission},
                "s": {"R": reflection.copy(), "T": transmission.copy()},
            },
        }
        metrics = circular.result_metrics(result, meta)
        self.assertTrue(metrics["valid"])
        self.assertEqual(metrics["max_pointwise_conservation_error"], 0.0)
        self.assertEqual(metrics["polarization_R_max_abs"], 0.0)
        self.assertEqual(metrics["polarization_T_max_abs"], 0.0)
        self.assertAlmostEqual(metrics["polarization_dE00"], 0.0)

        result["spectra"]["s"]["R"] = reflection + 1e-4
        result["spectra"]["s"]["T"] = transmission - 1e-4
        changed = circular.result_metrics(result, meta)
        self.assertGreater(changed["polarization_R_max_abs"], 1e-7)


if __name__ == "__main__":
    unittest.main()
