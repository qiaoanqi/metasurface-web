import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANIFEST = _load("make_ui_release_manifest_test", SCRIPTS / "make_ui_release_manifest.py")
VERIFY = _load("verify_ui_release_manifest_test", SCRIPTS / "verify_ui_release.py")


def test_manifest_is_stable_and_accepted_by_verify(tmp_path):
    (tmp_path / "a.py").write_text("value = 1\n", encoding="utf-8")
    whitelist = ("a.py",)
    first = MANIFEST.build_manifest(tmp_path, whitelist)
    second = MANIFEST.build_manifest(tmp_path, whitelist)
    assert first == second
    assert first["status"] == "pass"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(first), encoding="utf-8")
    result = VERIFY._check_manifest(tmp_path, manifest_path)
    assert result["status"] == "pass"


def test_manifest_detects_content_drift_through_verify(tmp_path):
    target = tmp_path / "tracked.txt"
    target.write_text("before", encoding="utf-8")
    report = MANIFEST.build_manifest(tmp_path, ("tracked.txt",))
    target.write_text("after", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(report), encoding="utf-8")
    result = VERIFY._check_manifest(tmp_path, manifest_path)
    assert result["status"] == "fail"
    assert result["mismatches"][0]["path"] == "tracked.txt"


def test_manifest_fails_closed_when_whitelisted_file_is_missing(tmp_path):
    result = MANIFEST.build_manifest(tmp_path, ("missing.py",))
    assert result["status"] == "fail"
    assert result["files"] == {}
    assert result["errors"][0]["path"] == "missing.py"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(result), encoding="utf-8")
    assert VERIFY._check_manifest(tmp_path, manifest_path)["status"] == "fail"


def test_manifest_rejects_whitelist_path_escape(tmp_path):
    result = MANIFEST.build_manifest(tmp_path, ("../outside.txt",))
    assert result["status"] == "fail"
    assert result["files"] == {}
    assert result["errors"][0]["reason"].startswith("ValueError:")


def test_manifest_json_has_no_absolute_paths_and_sorted_file_keys(tmp_path):
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    report = MANIFEST.build_manifest(tmp_path, ("z.txt", "a.txt"))
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)
    assert decoded == report
    assert list(report["files"]) == ["a.txt", "z.txt"]
    assert str(tmp_path) not in encoded
