import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_ui_release.py"
SPEC = importlib.util.spec_from_file_location("verify_ui_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_clean_project_gate_is_pass_and_json_stable():
    first = GATE.verify_release(ROOT)
    second = GATE.verify_release(ROOT)
    assert first == second
    assert first["schema"] == "ui-release-gate-v1"
    assert first["status"] == "pass"
    assert first["checks"][1]["name"] == "llm_features_disabled"
    assert first["checks"][1]["value"] is False
    assert first["checks"][-1]["status"] == "not_requested"
    # The report itself is JSON-serializable and has no machine-specific path.
    encoded = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert json.loads(encoded) == first
    assert first["root"] == "."


def test_manifest_hashes_accept_sha256_prefix_and_detect_drift(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("release", encoding="utf-8")
    digest = GATE._sha256(target)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": {"sample.txt": "sha256:" + digest.upper()}}),
        encoding="utf-8",
    )
    assert GATE._check_manifest(tmp_path, manifest)["status"] == "pass"
    target.write_text("drift", encoding="utf-8")
    result = GATE._check_manifest(tmp_path, manifest)
    assert result["status"] == "fail"
    assert result["mismatches"][0]["path"] == "sample.txt"


def test_manifest_rejects_path_escape_without_reading_outside_root(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"../outside": "0" * 64}), encoding="utf-8")
    result = GATE._check_manifest(tmp_path, manifest)
    assert result["status"] == "fail"
    assert result["mismatches"][0]["actual"].startswith("error:")


@pytest.mark.parametrize("raw", [{"files": []}, {"files": {"a": "bad"}}, []])
def test_manifest_shape_is_fail_closed(tmp_path, raw):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    assert GATE._check_manifest(tmp_path, manifest)["status"] == "fail"


def test_missing_llm_flag_fails_closed(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = False\n", encoding="utf-8")
    result = GATE._check_llm_flag(tmp_path)
    assert result["status"] == "fail"
