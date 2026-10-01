import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment: {name}")


def test_audited_demo_preset_is_bound_to_exact_reference_geometry():
    preset = _literal_assignment(ROOT / "app.py", "_AUDITED_REFERENCE_SAMPLE")
    assert preset == {
        "material": "TiO2 (anatase)",
        "substrate": "SiO2 (fused silica)",
        "polarization": "TM (p-pol)",
        "angle_deg": 0.0,
        "diameter_nm": 140.0,
        "height_nm": 281.0,
        "period_nm": 407.0,
        "target_hex": "#242d49",
    }


def test_local_delivery_docs_are_present_and_packaged():
    for relative in (
        "competition/11_本地离线演示说明.md",
        "competition/12_校赛本地交付清单.md",
    ):
        assert (ROOT / relative).is_file()
    build_script = (ROOT / "scripts/build_web_deployment_bundle.py").read_text(encoding="utf-8")
    assert "competition/11_本地离线演示说明.md" in build_script
    assert "competition/12_校赛本地交付清单.md" in build_script


def test_delivery_docs_keep_scientific_boundary_explicit():
    text = (ROOT / "competition/11_本地离线演示说明.md").read_text(encoding="utf-8")
    assert "不做最近邻、插值或训练" in text
    assert "不是全域收敛或模型准确率证明" in text


def test_bundle_builder_excludes_research_control_plane_and_history_archives():
    script = (ROOT / "scripts/build_web_deployment_bundle.py").read_text(encoding="utf-8")
    assert ".state/pool_manifest.json" not in script
    assert ".state/d65_colorimetry_v1_r2.json" not in script
    assert 'ignore=shutil.ignore_patterns("*.zip", "*.pptx")' in script
