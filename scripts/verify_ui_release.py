"""Read-only, deterministic release consistency gate for the UI project.

The gate deliberately avoids importing ``app.py``: importing the Streamlit
entrypoint can initialise optional services and is not required to verify a
release.  It compiles the entrypoint, imports only the small UI contract
modules, and can optionally compare hashes supplied by a caller-owned
manifest.

Exit status is 0 when all requested checks pass (``not_requested`` manifest
checks are informational), and 1 otherwise.  JSON output is stable so it can
be archived as a release evidence artifact.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile
from typing import Any, Mapping


GATE_SCHEMA = "ui-release-gate-v1"

# These files are the UI surface and its immutable scientific boundaries.  A
# manifest may narrow the hash set, but never changes this existence gate.
REQUIRED_FILES: tuple[str, ...] = (
    "app.py",
    "engine.py",
    "ml_module.py",
    "color_utils.py",
    "ui_forward_routes.py",
    "ui_analysis_snapshots.py",
    "ui_model_resources.py",
    "ui_pattern_contracts.py",
    "ui_inverse_contracts.py",
    "ui_session_migration.py",
    "ui_engine_session.py",
    "ui_fdtd_asset.py",
    "ui_cie_contracts.py",
    "pipeline_policy.json",
    ".state/pipeline_integrity.json",
)

COMPILE_FILES: tuple[str, ...] = tuple(
    path for path in REQUIRED_FILES if path.endswith(".py")
)

# Importing these modules is side-effect-light and exercises the symbols used
# by the UI.  app.py is intentionally checked by AST/py_compile only.
CONTRACT_IMPORTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ui_forward_routes", "ui_forward_routes.py", ("ForwardResult",)),
    (
        "ui_analysis_snapshots",
        "ui_analysis_snapshots.py",
        ("ANALYSIS_SNAPSHOT_SCHEMA_VERSION", "ANALYSIS_PROTOCOL_VERSION"),
    ),
    (
        "ui_model_resources",
        "ui_model_resources.py",
        ("BoundModelResource",),
    ),
    (
        "ui_pattern_contracts",
        "ui_pattern_contracts.py",
        (
            "PATTERN_SNAPSHOT_SCHEMA_VERSION",
            "PATTERN_PROTOCOL_VERSION",
            "PATTERN_CONTRACT_VERSION",
        ),
    ),
    (
        "ui_session_migration",
        "ui_session_migration.py",
        ("SESSION_MIGRATION_SCHEMA", "SESSION_MIGRATION_VERSION"),
    ),
    ("ui_engine_session", "ui_engine_session.py", ("ENGINE_SESSION_CONTRACT_VERSION",)),
    ("ui_fdtd_asset", "ui_fdtd_asset.py", ("FDTD_ASSET_PROTOCOL_VERSION",)),
)


def _result(name: str, status: str, **details: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"name": name, "status": status}
    value.update(details)
    return value


def _safe_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path while rejecting paths outside ``root``."""
    candidate = (root / relative).resolve()
    candidate.relative_to(root.resolve())
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ast_literal(tree: ast.AST, name: str) -> Any:
    for node in ast.walk(tree):
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == name and value is not None:
            return ast.literal_eval(value)
    raise ValueError(f"literal {name!r} not found")


def _check_llm_flag(root: Path) -> dict[str, Any]:
    path = root / "app.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        value = _ast_literal(tree, "ENABLE_LLM_FEATURES")
    except Exception as exc:  # deterministic class/message, no traceback in JSON
        return _result("llm_features_disabled", "fail", reason=f"{type(exc).__name__}: {exc}")
    if value is not False:
        return _result("llm_features_disabled", "fail", value=value)
    return _result("llm_features_disabled", "pass", value=False)


def _check_required_files(root: Path) -> dict[str, Any]:
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    return _result(
        "required_files",
        "pass" if not missing else "fail",
        checked=list(REQUIRED_FILES),
        missing=missing,
    )


def _check_compile(root: Path) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for relative in COMPILE_FILES:
        path = root / relative
        temporary_output: str | None = None
        try:
            # Explicit cfile keeps this read-only with respect to the project:
            # py_compile's default would create/update a repository __pycache__.
            with tempfile.NamedTemporaryFile(suffix=".pyc", delete=False) as handle:
                temporary_output = handle.name
            py_compile.compile(str(path), cfile=temporary_output, doraise=True)
        except Exception as exc:
            failures.append({"path": relative, "reason": f"{type(exc).__name__}: {exc}"})
        finally:
            if temporary_output:
                try:
                    os.unlink(temporary_output)
                except OSError:
                    pass
    return _result(
        "py_compile",
        "pass" if not failures else "fail",
        checked=list(COMPILE_FILES),
        failures=failures,
    )


def _import_contracts(root: Path) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    loaded: list[str] = []
    for module_name, relative, symbols in CONTRACT_IMPORTS:
        path = root / relative
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("module spec unavailable")
            module = importlib.util.module_from_spec(spec)
            # Contract modules use absolute local imports (for example
            # ``color_utils``), so resolve them against the project root.
            old_path = list(sys.path)
            old_module = sys.modules.get(module_name)
            sys.modules[module_name] = module
            sys.path.insert(0, str(root))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path[:] = old_path
                if old_module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = old_module
            missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
            if missing:
                raise AttributeError("missing symbols: " + ", ".join(missing))
            loaded.append(module_name)
        except Exception as exc:
            failures.append({"module": module_name, "reason": f"{type(exc).__name__}: {exc}"})
    return _result(
        "contract_imports",
        "pass" if not failures else "fail",
        checked=[item[0] for item in CONTRACT_IMPORTS],
        loaded=loaded,
        failures=failures,
    )


def _load_manifest(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    if value.get("status") == "fail":
        raise ValueError("manifest reports failed generation")
    files = value.get("files", value)
    if not isinstance(files, dict):
        raise ValueError("manifest files must be a JSON object")
    normalized: dict[str, str] = {}
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("manifest paths and hashes must be strings")
        digest = expected.lower()
        if digest.startswith("sha256:"):
            digest = digest[7:]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid sha256 for {relative}")
        normalized[relative.replace("\\", "/")] = digest
    return normalized


def _check_manifest(root: Path, manifest: Path | None) -> dict[str, Any]:
    if manifest is None:
        return _result("manifest_hashes", "not_requested", checked=[], mismatches=[])
    try:
        expected = _load_manifest(manifest)
    except Exception as exc:
        return _result("manifest_hashes", "fail", checked=[], mismatches=[], reason=f"{type(exc).__name__}: {exc}")
    mismatches: list[dict[str, str]] = []
    checked: list[str] = []
    for relative in sorted(expected):
        try:
            target = _safe_path(root, relative)
            actual = _sha256(target) if target.is_file() else "missing"
        except Exception as exc:
            actual = f"error:{type(exc).__name__}"
        checked.append(relative)
        if actual != expected[relative]:
            mismatches.append({"path": relative, "expected": expected[relative], "actual": actual})
    return _result(
        "manifest_hashes",
        "pass" if not mismatches else "fail",
        checked=checked,
        mismatches=mismatches,
    )


def verify_release(root: Path, manifest: Path | None = None) -> dict[str, Any]:
    """Return a deterministic JSON-compatible release report."""
    root = root.resolve()
    checks = [
        _check_required_files(root),
        _check_llm_flag(root),
        _check_compile(root),
        _import_contracts(root),
        _check_manifest(root, manifest),
    ]
    failed = any(check["status"] == "fail" for check in checks)
    return {
        "schema": GATE_SCHEMA,
        "root": ".",
        "status": "fail" if failed else "pass",
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    parser.add_argument("--manifest", type=Path, help="optional JSON path/hash manifest")
    parser.add_argument("--pretty", action="store_true", help="indent JSON for humans")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = args.manifest.resolve() if args.manifest else None
    report = verify_release(root, manifest)
    indent = 2 if args.pretty else None
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":") if indent is None else None, indent=indent))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
