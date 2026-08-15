import unittest

import numpy as np

import paper2_colorimetry as coarse
import paper2_colorimetry_fine as fine


class FineColorimetryTests(unittest.TestCase):
    def test_supported_grids_are_exact_and_include_both_endpoints(self):
        for step in fine.ALLOWED_STEPS_NM:
            wavelength = fine.wavelength_grid(step)
            self.assertEqual(wavelength[0], 380.0)
            self.assertEqual(wavelength[-1], 780.0)
            self.assertEqual(wavelength.size, int(round(400.0 / step)) + 1)
            self.assertEqual(fine.validate_wavelength_grid(wavelength), step)

    def test_perfect_reflector_is_neutral_on_every_grid(self):
        for step in fine.ALLOWED_STEPS_NM:
            wavelength = fine.wavelength_grid(step)
            labels = fine.spectrum_to_labels_d65(np.ones_like(wavelength), wavelength)
            np.testing.assert_allclose(labels["lab"], [100.0, 0.0, 0.0], atol=1e-10)

    def test_five_nm_path_matches_frozen_paper2_path(self):
        wavelength = fine.wavelength_grid(5.0)
        reflectance = np.linspace(0.02, 0.98, wavelength.size) ** 1.7
        expected = coarse.spectrum_to_labels_d65(reflectance, wavelength)
        actual = fine.spectrum_to_labels_d65(reflectance, wavelength)
        np.testing.assert_allclose(actual["xyz"], expected["xyz"], atol=1e-15)
        np.testing.assert_allclose(actual["lab"], expected["lab"], atol=1e-12)
        np.testing.assert_allclose(actual["srgb_display"], expected["srgb_display"], atol=1e-15)

    def test_batch_shape_is_preserved(self):
        wavelength = fine.wavelength_grid(1.0)
        spectra = np.stack([np.zeros_like(wavelength), np.ones_like(wavelength)])
        labels = fine.spectrum_to_labels_d65(spectra, wavelength)
        self.assertEqual(labels["xyz"].shape, (2, 3))
        self.assertEqual(labels["lab"].shape, (2, 3))

    def test_rejects_unregistered_grid_and_nonfinite_spectrum(self):
        with self.assertRaises(ValueError):
            fine.wavelength_grid(4.0)
        wavelength = np.arange(380.0, 781.0, 4.0)
        with self.assertRaises(ValueError):
            fine.spectrum_to_xyz_d65(np.ones_like(wavelength), wavelength)
        wavelength = fine.wavelength_grid(0.5)
        reflectance = np.ones_like(wavelength)
        reflectance[3] = np.nan
        with self.assertRaises(ValueError):
            fine.spectrum_to_xyz_d65(reflectance, wavelength)

    def test_provenance_discloses_interpolation(self):
        provenance = fine.label_provenance(0.25)
        self.assertEqual(provenance["wavelength_nm"]["samples"], 1601)
        self.assertIn("linear interpolation", provenance["resampling"])
        self.assertEqual(provenance["srgb_role"], "display_only_clipped")


if __name__ == "__main__":
    unittest.main()
