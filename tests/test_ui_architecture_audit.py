import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_ui_architecture.py"
SPEC = importlib.util.spec_from_file_location("audit_ui_architecture_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_project_architecture_report_is_stable_and_passes():
    first = AUDIT.audit_architecture(ROOT)
    second = AUDIT.audit_architecture(ROOT)
    assert first == second
    assert first["schema"] == "ui-architecture-audit-v1"
    assert first["status"] == "pass"
    assert first["target"] == "app.py"
    assert first["stats"]["line_count"] > 1000
    assert first["stats"]["top_level_functions"] > 50
    assert first["stats"]["top_level_imports"] > 10
    assert "rcwa_surrogate" in first["route_literals"]
    assert first["llm_features_disabled"]["value"] is False
    assert first["startup_boundary"]["violations"] == []
    assert json.loads(json.dumps(first, ensure_ascii=False, sort_keys=True)) == first


def test_temporary_app_with_network_and_runner_start_fails(tmp_path):
    source = """
import requests
from runner import start_runner
ENABLE_LLM_FEATURES = True
requests.get('https://example.invalid')
start_runner()
def helper():
    import urllib.request
    return urllib.request.urlopen('https://example.invalid')
"""
    (tmp_path / "bad.py").write_text(source, encoding="utf-8")
    report = AUDIT.audit_architecture(tmp_path, Path("bad.py"))
    assert report["status"] == "fail"
    assert report["llm_features_disabled"]["value"] is True
    assert "requests" in report["startup_boundary"]["forbidden_imports"]
    assert "requests.get" in report["startup_boundary"]["forbidden_calls"]
    assert "start_runner" in report["startup_boundary"]["top_level_runner_calls"]


def test_path_escape_fails_closed(tmp_path):
    report = AUDIT.audit_architecture(tmp_path, Path("../outside.py"))
    assert report == {
        "schema": "ui-architecture-audit-v1",
        "status": "fail",
        "target": "../outside.py",
        "error": "path_outside_root",
    }


def test_non_python_and_missing_targets_fail_closed(tmp_path):
    (tmp_path / "notes.txt").write_text("not code", encoding="utf-8")
    non_python = AUDIT.audit_architecture(tmp_path, Path("notes.txt"))
    missing = AUDIT.audit_architecture(tmp_path, Path("missing.py"))
    assert non_python["status"] == "fail"
    assert non_python["error"] == "target_not_python"
    assert missing["status"] == "fail"
    assert missing["error"] == "target_missing"


def test_syntax_error_is_deterministic_and_fail_closed(tmp_path):
    (tmp_path / "broken.py").write_text("if :\n", encoding="utf-8")
    first = AUDIT.audit_architecture(tmp_path, Path("broken.py"))
    second = AUDIT.audit_architecture(tmp_path, Path("broken.py"))
    assert first == second
    assert first["status"] == "fail"
    assert first["error"].startswith("SyntaxError:")
