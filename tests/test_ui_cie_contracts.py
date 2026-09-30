import ast
from pathlib import Path

import numpy as np

from ui_cie_contracts import (
    GamutSamples,
    build_cie_plot_data,
    current_cie_point,
    evaluate_gamut,
    forward_provenance_caption,
)
from ui_forward_routes import (
    ForwardResult,
    make_result_provenance,
    normalize_forward_result,
)


def _provenance(*, material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    return make_result_provenance(
        "fp_tmm", "FP cavity TMM", ["FP TMM", "CIE D65 colorimetry"],
        "薄膜腔 TMM", "none", "fp_cavity.py", material, substrate,
        "TE (s-pol)", 15.0, "未启用", 0.0,
    )


def _available_forward(provenance=None):
    wavelengths = np.linspace(380.0, 780.0, 81)
    reflectance = np.linspace(0.1, 0.8, 81)
    return normalize_forward_result(
        wavelengths, reflectance, provenance or _provenance())


def test_unavailable_forward_has_no_current_point_or_fake_xy(monkeypatch):
    import color_utils

    calls = {"rgb_to_xy": 0}

    def forbidden_rgb_to_xy(*_args, **_kwargs):
        calls["rgb_to_xy"] += 1
        raise AssertionError("rgb_to_xy must not be called for an unavailable result")

    monkeypatch.setattr(color_utils, "rgb_to_xy", forbidden_rgb_to_xy)
    forward = ForwardResult(
        None, None, None, _provenance(), False, "model output unavailable")
    point = current_cie_point(forward)
    plot = build_cie_plot_data(
        forward,
        np.array([0.1, 0.2, 0.3]),
        np.array([0.2, 0.3, 0.2]),
        np.array([0.7, 0.5, 0.5]),
    )

    assert point.available is False
    assert point.xy is None
    assert point.rgb is None
    assert "unavailable" in point.reason
    assert plot.current.xy is None
    assert calls == {"rgb_to_xy": 0}


def test_available_point_comes_from_the_canonical_spectrum():
    forward = _available_forward()
    point = current_cie_point(forward)

    assert point.available is True
    assert point.xy is not None
    assert len(point.xy) == 2
    assert np.all(np.isfinite(point.xy))
    assert np.array_equal(point.rgb, forward.rgb)


def test_fp_dbr_and_ag_captions_use_only_canonical_provenance():
    dbr = forward_provenance_caption(_provenance(
        material="TiO2 (cavity layer)",
        substrate="SiO2 (DBR mirror stack)",
    ))
    ag = forward_provenance_caption(_provenance(
        material="TiO2 (cavity layer)",
        substrate="Ag (metal mirrors)",
    ))

    assert "材料：TiO2 (cavity layer)" in dbr
    assert "衬底：SiO2 (DBR mirror stack)" in dbr
    assert "材料：TiO2 (cavity layer)" in ag
    assert "衬底：Ag (metal mirrors)" in ag
    for caption in (dbr, ag):
        assert "偏振：TE (s-pol)" in caption
        assert "入射角：15 deg" in caption
        assert "路线：FP cavity TMM" in caption
        assert "模型：fp_cavity.py" in caption


def test_caption_does_not_backfill_missing_structure_from_external_state():
    provenance = _provenance()
    assert provenance["structure_type"] == "unknown"
    assert provenance["geometry_summary"] == "unknown"

    caption = forward_provenance_caption(provenance)

    assert "结构：unknown" in caption
    assert "几何：unknown" in caption
    assert "D=" not in caption


def test_gamut_failure_and_partial_failure_are_explicit():
    failed = evaluate_gamut(
        "Lorentz/Fano analytical approximation", "TiO2 / SiO2",
        lambda: (_ for _ in ()).throw(RuntimeError("route failed")),
    )
    partial = evaluate_gamut(
        "FP-TMM", "TiO2 cavity / Ag mirrors",
        lambda: GamutSamples(
            np.array([[0.2, 0.3], [0.3, 0.4], [0.4, 0.2]]), 2),
    )

    assert failed.available is False
    assert failed.points_xy.shape == (0, 2)
    assert "RuntimeError" in failed.error
    assert partial.available is True
    assert "2 个采样点计算失败" in partial.warning


def test_cie_gamut_source_has_no_fake_engine_or_bare_except():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "MetaEngine" not in source
    assert "_physical_gamut_xy" not in source
    assert "Lorentz/Fano analytical approximation" in source
    assert "不是直接 RCWA、代理模型精度或实验色域" in source
    assert "当前 CIE 点不可用，未绘制占位点" in source
    assert "canonical JSON SHA-256" in Path(
        "ui_model_difference_contracts.py").read_text(encoding="utf-8")

    affected = source[
        source.index("# Tab 5: Spectrum"):
        source.index("    def _benchmark_methods()")
    ]
    tree = ast.parse(affected)
    assert all(handler.type is not None for handler in (
        node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
    ))
    assert "st.warning(" in affected
    assert "st.error(" in affected
