"""Fine-grid D65 colorimetry for versioned paper 2 replacement pools.

The frozen paper 2 D65/observer tables are sampled at 5 nm.  Fine solver
grids use deterministic linear interpolation of those tables before numerical
integration.  This module records that interpolation explicitly; it does not
claim an independent sub-nanometre tabulation of the CIE standards.
"""

from __future__ import annotations

import numpy as np

from color_utils import CIE_X, CIE_Y, CIE_Z, SRGB_M
from paper2_colorimetry import D65_SPD, WAVELENGTHS_NM


COLORIMETRY_VERSION = "paper2-d65-fine-v1"
ALLOWED_STEPS_NM = (0.25, 0.5, 1.0, 2.0, 2.5, 5.0)
_CMF_5NM = np.stack([CIE_X, CIE_Y, CIE_Z], axis=-1).astype(np.float64)


def wavelength_grid(step_nm: float) -> np.ndarray:
    """Return an exact supported uniform grid from 380 through 780 nm."""
    step = float(step_nm)
    if step not in ALLOWED_STEPS_NM:
        raise ValueError(f"unsupported wavelength step {step_nm!r} nm")
    samples = int(round(400.0 / step)) + 1
    grid = np.linspace(380.0, 780.0, samples, dtype=np.float64)
    if not np.allclose(np.diff(grid), step, rtol=0.0, atol=1e-12):
        raise RuntimeError("failed to construct the requested wavelength grid")
    return grid


def validate_wavelength_grid(wavelengths_nm: np.ndarray) -> float:
    """Validate a supported exact grid and return its step in nanometres."""
    wavelength = np.asarray(wavelengths_nm, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("wavelength grid must be a one-dimensional array")
    for step in ALLOWED_STEPS_NM:
        expected = wavelength_grid(step)
        if wavelength.shape == expected.shape and np.array_equal(wavelength, expected):
            return step
    raise ValueError("grid must be exactly 380-780 nm at a registered paper 2 step")


def observer_and_illuminant(wavelengths_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the frozen 5 nm observer and D65 tables onto a valid grid."""
    wavelength = np.asarray(wavelengths_nm, dtype=np.float64)
    validate_wavelength_grid(wavelength)
    cmf = np.stack(
        [np.interp(wavelength, WAVELENGTHS_NM, _CMF_5NM[:, index]) for index in range(3)],
        axis=-1,
    )
    illuminant = np.interp(wavelength, WAVELENGTHS_NM, D65_SPD)
    return cmf, illuminant


def white_xyz_d65(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Return the numerical D65 white for the supplied supported grid."""
    wavelength = np.asarray(wavelengths_nm, dtype=np.float64)
    cmf, illuminant = observer_and_illuminant(wavelength)
    normalization = float(np.trapezoid(illuminant * cmf[:, 1], wavelength))
    return np.trapezoid(illuminant[:, None] * cmf, wavelength, axis=0) / normalization


def spectrum_to_xyz_d65(
    reflectance: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Integrate one or more reflectance spectra into unclipped D65 XYZ."""
    wavelength = np.asarray(wavelengths_nm, dtype=np.float64)
    cmf, illuminant = observer_and_illuminant(wavelength)
    reflectance_array = np.asarray(reflectance, dtype=np.float64)
    if reflectance_array.shape[-1:] != (wavelength.size,):
        raise ValueError("reflectance final axis must match the wavelength grid")
    if not np.isfinite(reflectance_array).all():
        raise ValueError("reflectance contains non-finite values")
    normalization = float(np.trapezoid(illuminant * cmf[:, 1], wavelength))
    weighted = reflectance_array[..., :, None] * illuminant[:, None] * cmf
    return np.trapezoid(weighted, wavelength, axis=-2) / normalization


def xyz_to_lab_d65(xyz: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Convert unclipped XYZ directly to CIELAB on the grid's numerical white."""
    xyz_array = np.asarray(xyz, dtype=np.float64)
    if xyz_array.shape[-1:] != (3,) or not np.isfinite(xyz_array).all():
        raise ValueError("xyz must be finite with three values on its final axis")
    ratio = xyz_array / white_xyz_d65(wavelengths_nm)
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    transformed = np.where(
        ratio > epsilon,
        np.cbrt(ratio),
        (kappa * ratio + 16.0) / 116.0,
    )
    return np.stack(
        [
            116.0 * transformed[..., 1] - 16.0,
            500.0 * (transformed[..., 0] - transformed[..., 1]),
            200.0 * (transformed[..., 1] - transformed[..., 2]),
        ],
        axis=-1,
    )


def xyz_to_srgb_display(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to clipped sRGB for display only."""
    xyz_array = np.asarray(xyz, dtype=np.float64)
    if xyz_array.shape[-1:] != (3,) or not np.isfinite(xyz_array).all():
        raise ValueError("xyz must be finite with three values on its final axis")
    linear = np.clip(xyz_array @ SRGB_M.T, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def spectrum_to_labels_d65(
    reflectance: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Produce XYZ/Lab labels and a display-only clipped sRGB value."""
    xyz = spectrum_to_xyz_d65(reflectance, wavelengths_nm)
    return {
        "xyz": xyz,
        "lab": xyz_to_lab_d65(xyz, wavelengths_nm),
        "srgb_display": xyz_to_srgb_display(xyz),
    }


def label_provenance(step_nm: float) -> dict[str, object]:
    """Return serializable provenance for a supported fine-grid label set."""
    wavelength = wavelength_grid(step_nm)
    return {
        "version": COLORIMETRY_VERSION,
        "illuminant": "CIE standard illuminant D65",
        "observer": "CIE 1931 2 degree standard observer",
        "source_tables": "paper2-d65-v1 frozen 5 nm D65 and observer arrays",
        "resampling": "deterministic linear interpolation onto solver grid",
        "wavelength_nm": {
            "start": 380.0,
            "stop": 780.0,
            "step": float(step_nm),
            "samples": int(wavelength.size),
        },
        "normalization": "k = 1 / integral(S_D65 * ybar dlambda)",
        "lab_source": "direct_unclipped_xyz",
        "srgb_role": "display_only_clipped",
        "white_xyz": white_xyz_d65(wavelength).tolist(),
    }
