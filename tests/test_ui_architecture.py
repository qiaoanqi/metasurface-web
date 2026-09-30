"""Named release-facing entrypoint for the architecture audit smoke test."""
from pathlib import Path

from test_ui_architecture_audit import AUDIT


ROOT = Path(__file__).resolve().parents[1]


def test_named_architecture_audit_entrypoint_is_healthy():
    report = AUDIT.audit_architecture(ROOT)
    assert report["schema"] == "ui-architecture-audit-v1"
    assert report["status"] == "pass"
    assert report["startup_boundary"]["violations"] == []
