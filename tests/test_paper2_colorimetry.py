import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from paper2_colorimetry import (
    WAVELENGTHS_NM,
    d65_white_xyz,
    label_provenance,
    spectrum_to_labels_d65,
    spectrum_to_xyz_d65,
    xyz_to_lab_d65,
)
from pipeline_supervisor import atomic_json, file_digest


class Paper2D65ColorimetryTests(unittest.TestCase):
    def test_perfect_reflector_is_neutral_d65_white(self):
        labels = spectrum_to_labels_d65(np.ones(81))
        np.testing.assert_allclose(labels["xyz"], d65_white_xyz(), atol=1e-12, rtol=0)
        np.testing.assert_allclose(labels["lab"], [100.0, 0.0, 0.0], atol=1e-10, rtol=0)
        xy = labels["xyz"][:2] / np.sum(labels["xyz"])
        np.testing.assert_allclose(xy, [0.3127, 0.3290], atol=5e-4, rtol=0)
        np.testing.assert_allclose(labels["srgb_display"], [1.0, 1.0, 1.0], atol=2e-3, rtol=0)

    def test_black_reflector_is_zero(self):
        labels = spectrum_to_labels_d65(np.zeros(81))
        np.testing.assert_allclose(labels["xyz"], [0.0, 0.0, 0.0], atol=0, rtol=0)
        np.testing.assert_allclose(labels["lab"], [0.0, 0.0, 0.0], atol=1e-12, rtol=0)

    def test_lab_is_computed_from_unclipped_xyz(self):
        reflectance = np.zeros(81)
        reflectance[(WAVELENGTHS_NM >= 440) & (WAVELENGTHS_NM <= 460)] = 1.0
        labels = spectrum_to_labels_d65(reflectance)
        np.testing.assert_allclose(labels["lab"], xyz_to_lab_d65(labels["xyz"]), atol=0, rtol=0)
        self.assertTrue(np.any((labels["srgb_display"] == 0.0) | (labels["srgb_display"] == 1.0)))

    def test_batch_and_scalar_paths_match(self):
        spectra = np.stack([np.zeros(81), np.ones(81), np.linspace(0.0, 1.0, 81)])
        batch = spectrum_to_xyz_d65(spectra)
        scalar = np.stack([spectrum_to_xyz_d65(row) for row in spectra])
        np.testing.assert_allclose(batch, scalar, atol=1e-14, rtol=0)

    def test_grid_and_provenance_are_explicit(self):
        with self.assertRaises(ValueError):
            spectrum_to_xyz_d65(np.ones(80), np.arange(380.0, 780.0, 5.0))
        provenance = label_provenance()
        self.assertEqual(provenance["version"], "paper2-d65-v1")
        self.assertEqual(provenance["lab_source"], "direct_unclipped_xyz")
        self.assertEqual(provenance["srgb_role"], "display_only_clipped")

    def test_evidence_json_writer_is_content_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            payload = {"schema_version": 1, "passed": True, "generated_at": "fixed"}
            atomic_json(path, payload)
            first = file_digest(path)
            existing = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(existing, payload)
            atomic_json(path, existing)
            self.assertEqual(file_digest(path), first)


if __name__ == "__main__":
    unittest.main()
