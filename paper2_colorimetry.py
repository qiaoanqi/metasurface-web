"""Versioned D65 colorimetry for paper 2 derived labels.

This module is intentionally separate from color_utils.py, which remains the
legacy paper 1 path. Lab is derived directly from unclipped XYZ. Clipped sRGB
is produced only as a display value.
"""

from __future__ import annotations

import numpy as np

from color_utils import CIE_X, CIE_Y, CIE_Z, SRGB_M


COLORIMETRY_VERSION = "paper2-d65-v1"
ILLUMINANT_SOURCE = {
    "name": "CIE standard illuminant D65",
    "doi": "10.25039/CIE.DS.hjfjmt59",
    "source_grid_nm": "300-830 nm at 1 nm",
    "license": "CC BY-SA 4.0",
}
OBSERVER = "CIE 1931 2 degree standard observer"
WAVELENGTHS_NM = np.arange(380.0, 785.0, 5.0, dtype=np.float64)

# Exact 5 nm samples from the CIE 1 nm D65 dataset above.
D65_SPD = np.array([
    49.97550, 52.31180, 54.64820, 68.70150, 82.75490, 87.12040,
    91.48600, 92.45890, 93.43180, 90.05700, 86.68230, 95.77360,
    104.86500, 110.93600, 117.00800, 117.41000, 117.81200, 116.33600,
    114.86100, 115.39200, 115.92300, 112.36700, 108.81100, 109.08200,
    109.35400, 108.57800, 107.80200, 106.29600, 104.79000, 106.23900,
    107.68900, 106.04700, 104.40500, 104.22500, 104.04600, 102.02300,
    100.00000, 98.16710, 96.33420, 96.06110, 95.78800, 92.23680,
    88.68560, 89.34590, 90.00620, 89.80260, 89.59910, 88.64890,
    87.69870, 85.49360, 83.28860, 83.49390, 83.69920, 81.86300,
    80.02680, 80.12070, 80.21460, 81.24620, 82.27780, 80.28100,
    78.28420, 74.00270, 69.72130, 70.66520, 71.60910, 72.97900,
    74.34900, 67.97650, 61.60400, 65.74480, 69.88560, 72.48630,
    75.08700, 69.33980, 63.59270, 55.00540, 46.41820, 56.61180,
    66.80540, 65.09410, 63.38280,
], dtype=np.float64)

_CMF = np.stack([CIE_X, CIE_Y, CIE_Z], axis=-1).astype(np.float64)
_NORMALIZATION = float(np.trapezoid(D65_SPD * CIE_Y, WAVELENGTHS_NM))
_WHITE_XYZ = (
    np.trapezoid(D65_SPD[:, None] * _CMF, WAVELENGTHS_NM, axis=0)
    / _NORMALIZATION
)


def _validate_grid(wavelengths_nm: np.ndarray) -> None:
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    if wavelengths_nm.shape != WAVELENGTHS_NM.shape or not np.array_equal(
        wavelengths_nm, WAVELENGTHS_NM
    ):
        raise ValueError("paper2-d65-v1 requires the exact 380-780 nm, 5 nm grid")


def d65_white_xyz() -> np.ndarray:
    """Return the numerical white used by this exact grid and integration."""
    return _WHITE_XYZ.copy()


def spectrum_to_xyz_d65(
    reflectance: np.ndarray,
    wavelengths_nm: np.ndarray = WAVELENGTHS_NM,
) -> np.ndarray:
    """Convert reflectance to D65-weighted XYZ without clipping the spectrum."""
    _validate_grid(wavelengths_nm)
    reflectance = np.asarray(reflectance, dtype=np.float64)
    if reflectance.shape[-1:] != (WAVELENGTHS_NM.size,):
        raise ValueError("reflectance must have 81 samples on its final axis")
    if not np.isfinite(reflectance).all():
        raise ValueError("reflectance contains non-finite values")
    weighted = reflectance[..., :, None] * D65_SPD[:, None] * _CMF
    return np.trapezoid(weighted, WAVELENGTHS_NM, axis=-2) / _NORMALIZATION


def xyz_to_lab_d65(xyz: np.ndarray) -> np.ndarray:
    """Convert unclipped XYZ directly to CIELAB using the numerical D65 white."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[-1:] != (3,) or not np.isfinite(xyz).all():
        raise ValueError("xyz must be finite with three values on its final axis")
    ratio = xyz / _WHITE_XYZ
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(ratio > epsilon, np.cbrt(ratio), (kappa * ratio + 16.0) / 116.0)
    return np.stack(
        [116.0 * f[..., 1] - 16.0,
         500.0 * (f[..., 0] - f[..., 1]),
         200.0 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def xyz_to_srgb_display(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to clipped sRGB for display; never use this output for Lab."""
    xyz = np.asarray(xyz, dtype=np.float64)
    linear = np.clip(xyz @ SRGB_M.T, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def spectrum_to_labels_d65(
    reflectance: np.ndarray,
    wavelengths_nm: np.ndarray = WAVELENGTHS_NM,
) -> dict[str, np.ndarray]:
    """Produce paper 2 XYZ/Lab labels and a display-only sRGB value."""
    xyz = spectrum_to_xyz_d65(reflectance, wavelengths_nm)
    return {
        "xyz": xyz,
        "lab": xyz_to_lab_d65(xyz),
        "srgb_display": xyz_to_srgb_display(xyz),
    }


def label_provenance() -> dict[str, object]:
    """Return serializable provenance required for any derived label artifact."""
    return {
        "version": COLORIMETRY_VERSION,
        "illuminant": ILLUMINANT_SOURCE,
        "observer": OBSERVER,
        "wavelength_nm": {"start": 380, "stop": 780, "step": 5, "samples": 81},
        "normalization": "k = 1 / integral(S_D65 * ybar dlambda)",
        "lab_source": "direct_unclipped_xyz",
        "srgb_role": "display_only_clipped",
        "white_xyz": _WHITE_XYZ.tolist(),
    }
