#!/usr/bin/env python3
"""Idempotent local controller for the paper 2 pipeline.

The controller observes producer state, validates immutable artifacts, reconciles
stale producer status from disk evidence, and emits a durable dispatch request.
It never edits data pools, paper files, or training code.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".state"
STATUS = STATE / "hermes_status.json"
POLICY = ROOT / "pipeline_policy.json"
AUDIT_RESULT = STATE / "audit_result.json"
NEXT_PLAN = STATE / "next_plan.json"
CONTROLLER_STATE = STATE / "controller_state.json"
DISPATCH_REQUEST = STATE / "dispatch_request.json"
EXECUTOR_ACK = STATE / "executor_ack.json"
LEGACY_INBOX = STATE / "hermes_inbox.json"
GATE_STATE = STATE / "gate_state.json"
SUPERVISOR_LOCK = STATE / "pipeline_supervisor.lock"
ATOMIC_REPLACE_ATTEMPTS = 10
REFERENCE_HOLDOUT_FINAL_REFERENCE = {
    "requested_nG": 450,
    "Nxy": 768,
    "wavelength_step_nm": 0.5,
}
AUDITOR_RUNTIME_PATHS = {
    "replacement_pool_generation": {
        "pipeline_supervisor.py",
        "scripts/audit_replacement_pool.py",
        "scripts/run_replacement_pool.py",
        "rcwa_batch.py",
        "paper2_colorimetry_fine.py",
        "color_utils.py",
    },
    "joint_numerical_convergence": {
        "pipeline_supervisor.py",
        "scripts/audit_joint_convergence_v2.py",
        "scripts/run_joint_convergence_v2.py",
        "scripts/run_replacement_pool.py",
        "scripts/run_reference_resolution_escalation.py",
        "scripts/run_reference_resolution_holdout.py",
        "scripts/run_reference_resolution_budget_v2.py",
        "scripts/freeze_reference_holdout_plan.py",
        "scripts/freeze_reference_budget_v2.py",
        "scripts/reference_protocol_selection.py",
        "scripts/reference_v1_outcome.py",
        "scripts/run_joint_convergence.py",
        "rcwa_batch.py",
        "paper2_colorimetry_fine.py",
        "color_utils.py",
    },
    "cross_solver_spectrum_validation": {
        "pipeline_supervisor.py",
        "scripts/audit_cross_solver_v2.py",
        "scripts/run_cross_solver_validation_v2.py",
        "scripts/run_cross_solver_validation.py",
        "scripts/run_joint_convergence_v2.py",
        "scripts/run_joint_convergence.py",
        "scripts/run_replacement_pool.py",
        "scripts/run_reference_resolution_escalation.py",
        "scripts/run_reference_resolution_holdout.py",
        "scripts/run_reference_resolution_budget_v2.py",
        "scripts/freeze_reference_holdout_plan.py",
        "scripts/freeze_reference_budget_v2.py",
        "scripts/reference_protocol_selection.py",
        "scripts/reference_v1_outcome.py",
        "rcwa_batch.py",
        "paper2_colorimetry.py",
        "paper2_colorimetry_fine.py",
        "color_utils.py",
    },
    "reference_resolution": {
        "pipeline_supervisor.py",
        "scripts/audit_reference_resolution_holdout.py",
        "scripts/run_reference_resolution_holdout.py",
        "scripts/run_reference_resolution_budget_v2.py",
        "scripts/run_reference_resolution_escalation.py",
        "scripts/freeze_reference_holdout_plan.py",
        "scripts/freeze_reference_budget_v2.py",
        "scripts/reference_protocol_selection.py",
        "scripts/reference_v1_outcome.py",
        "scripts/run_replacement_pool.py",
        "scripts/run_joint_convergence.py",
        "rcwa_batch.py",
        "paper2_colorimetry_fine.py",
        "color_utils.py",
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a process-specific temporary name so two independent writers cannot
    # replace or clean up each other's state file on Windows.
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    text = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    try:
        with tmp.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except OSError as exc:
                # Windows readers, antivirus scanners, and indexers can briefly
                # hold the destination without allowing replacement.
                transient = getattr(exc, "winerror", None) in {5, 32} or exc.errno in {13, 16}
                if not transient or attempt == ATOMIC_REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_payload_digest(payload: Any) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def production_reference_audit_approved(audit: Any) -> bool:
    return bool(
        isinstance(audit, dict)
        and audit.get("evidence_version") == "paper2-reference-holdout-audit-v1"
        and audit.get("protocol_revision") == "v2_bound_holdout"
        and audit.get("classification") == "reference_holdout_passed"
        and audit.get("final_reference") == REFERENCE_HOLDOUT_FINAL_REFERENCE
        and audit.get("passed") is True
        and audit.get("production_reference_approved") is True
    )


def run_auto_transition(controller: dict[str, Any]) -> dict[str, Any] | None:
    """Run a registered, deterministic post-failure transition helper."""
    dispatch = controller.get("dispatch")
    terminal_scientific = bool(
        isinstance(dispatch, dict)
        and dispatch.get("status") == "failed"
        and dispatch.get("terminal_failure") is True
        and str(dispatch.get("failure_class", "")).lower() == "scientific"
    )
    if not terminal_scientific:
        return None
    action = dispatch.get("action")
    if action not in {"joint_numerical_convergence", "replacement_pool_generation"}:
        return None

    steps: list[list[str]]
    if action == "replacement_pool_generation":
        audit_path = STATE / "replacement_pool_v1_audit.json"
        steps = [
            [sys.executable, str(ROOT / "scripts" / "audit_replacement_pool.py")],
            [
                sys.executable,
                str(ROOT / "scripts" / "activate_replacement_pool.py"),
                "--audit",
                str(audit_path.relative_to(ROOT)),
            ],
        ]
    else:
        steps = [[sys.executable, str(ROOT / "scripts" / "paper2_auto_transition.py")]]

    completed = None
    for command in steps:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"paper2 auto-transition failed: {detail}")
    assert completed is not None
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("paper2 auto-transition returned no status")
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("paper2 auto-transition returned invalid status")
    if action == "replacement_pool_generation":
        if result.get("active") is not True or not result.get("pool_sha256"):
            raise RuntimeError("replacement activation did not return a hash-bound active pool")
        return {
            "status": "advanced",
            "transition": "replacement_pool_activation",
            "based_on_request_id": dispatch.get("request_id"),
            "independent_audit": {
                "path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": file_digest(audit_path),
            },
            **result,
        }
    return result


def pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code)
            ):
                return False
            return exit_code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_single_instance_lock(path: Path):
    """Hold an OS-backed byte lock for the lifetime of a watch process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def load_policy(path: Path = POLICY) -> dict[str, Any]:
    policy = load_json(path)
    if not isinstance(policy, dict):
        raise ValueError(f"invalid or missing policy: {path}")
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported pipeline policy schema")
    return policy


def add_error(errors: list[dict[str, Any]], code: str, **details: Any) -> None:
    errors.append({"code": code, **details})


def audit_pool(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not path.exists():
        return {
            "path": str(path),
            "passed": False,
            "healthy_checkpoint": False,
            "records": 0,
            "errors": [{"code": "OUTPUT_MISSING"}],
            "warnings": [],
        }
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        return {
            "path": str(path),
            "passed": False,
            "healthy_checkpoint": False,
            "records": 0,
            "errors": [{"code": "PICKLE_UNREADABLE", "detail": str(exc)}],
            "warnings": [],
        }

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        records = []
        add_error(errors, "RECORDS_NOT_LIST")

    expected_records = int(spec["expected_records"])
    expected_wl = np.asarray(spec["wavelength_nm"], dtype=float)
    required_fields = set(spec["required_record_fields"])
    polarizations = set(spec["polarizations"])
    range_tol = float(spec.get("range_tolerance", 1e-8))
    conservation_tol = float(spec.get("pointwise_conservation_tolerance", 1e-6))
    stored_tol = float(spec.get("stored_value_tolerance", 1e-9))
    quality_tol = float(spec.get("quality_tolerance", 0.05))

    keys: list[tuple[Any, ...]] = []
    geometry_pols: dict[tuple[Any, ...], set[str]] = {}
    rt_means: list[float] = []
    pointwise_error_max = 0.0
    extrema = {"R_min": math.inf, "R_max": -math.inf, "T_min": math.inf, "T_max": -math.inf}
    long_axis_reversed = 0

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            add_error(errors, "RECORD_NOT_OBJECT", index=index)
            continue
        missing = sorted(required_fields - set(record))
        if missing:
            add_error(errors, "MISSING_FIELDS", index=index, fields=missing)
            continue
        if not record.get("success"):
            add_error(errors, "FAILED_RECORD", index=index)
            continue
        try:
            L, W, H, P = (float(record[name]) for name in ("L", "W", "H", "P"))
            pol = str(record["pol"])
            wl = np.asarray(record["wl_nm"], dtype=float)
            refl = np.asarray(record["R"], dtype=float)
            tran = np.asarray(record["T"], dtype=float)
        except Exception as exc:
            add_error(errors, "RECORD_PARSE_ERROR", index=index, detail=str(exc))
            continue

        if pol not in polarizations:
            add_error(errors, "BAD_POLARIZATION", index=index, value=pol)
        if wl.shape != expected_wl.shape or not np.allclose(wl, expected_wl, atol=0, rtol=0):
            add_error(errors, "BAD_WAVELENGTH_GRID", index=index, shape=list(wl.shape))
        if refl.shape != expected_wl.shape or tran.shape != expected_wl.shape:
            add_error(
                errors,
                "BAD_SPECTRUM_SHAPE",
                index=index,
                R_shape=list(refl.shape),
                T_shape=list(tran.shape),
            )
            continue
        if not (np.isfinite(refl).all() and np.isfinite(tran).all()):
            add_error(errors, "NONFINITE_SPECTRUM", index=index)
            continue

        local_extrema = {
            "R_min": float(np.min(refl)),
            "R_max": float(np.max(refl)),
            "T_min": float(np.min(tran)),
            "T_max": float(np.max(tran)),
        }
        extrema["R_min"] = min(extrema["R_min"], local_extrema["R_min"])
        extrema["R_max"] = max(extrema["R_max"], local_extrema["R_max"])
        extrema["T_min"] = min(extrema["T_min"], local_extrema["T_min"])
        extrema["T_max"] = max(extrema["T_max"], local_extrema["T_max"])
        if (
            local_extrema["R_min"] < -range_tol
            or local_extrema["R_max"] > 1.0 + range_tol
            or local_extrema["T_min"] < -range_tol
            or local_extrema["T_max"] > 1.0 + range_tol
        ):
            add_error(errors, "SPECTRUM_OUT_OF_RANGE", index=index)

        local_pointwise_max = float(np.max(np.abs(refl + tran - 1.0)))
        pointwise_error_max = max(pointwise_error_max, local_pointwise_max)
        if spec.get("lossless") and local_pointwise_max > conservation_tol:
            add_error(
                errors,
                "POINTWISE_CONSERVATION_FAIL",
                index=index,
                max_abs_error=local_pointwise_max,
            )

        rt_mean = float(np.mean(refl + tran))
        rt_means.append(rt_mean)
        if abs(float(record["R_plus_T_mean"]) - rt_mean) > stored_tol:
            add_error(errors, "STORED_RT_MISMATCH", index=index)
        expected_quality = abs(rt_mean - 1.0) <= quality_tol
        if bool(record["quality_pass"]) != expected_quality:
            add_error(errors, "QUALITY_FLAG_MISMATCH", index=index)
        if not expected_quality:
            add_error(errors, "QUALITY_FAIL", index=index, R_plus_T_mean=rt_mean)

        if not (80.0 <= L <= 350.0 and 80.0 <= W <= 350.0):
            add_error(errors, "GEOMETRY_AXIS_OUT_OF_RANGE", index=index)
        if not (100.0 <= H <= 600.0 and 200.0 <= P <= 600.0):
            add_error(errors, "GEOMETRY_HP_OUT_OF_RANGE", index=index)
        if max(L, W) >= P:
            add_error(errors, "GEOMETRY_OVERLAP", index=index)
        fill = math.pi * (L / 2.0) * (W / 2.0) / (P * P)
        aspect = max(L, W) / min(L, W)
        if not (0.03 - 1e-12 <= fill <= 0.70 + 1e-12):
            add_error(errors, "FILL_FRACTION_OUT_OF_RANGE", index=index, value=fill)
        if not (1.0 <= aspect <= 3.0 + 1e-12):
            add_error(errors, "ASPECT_RATIO_OUT_OF_RANGE", index=index, value=aspect)
        if abs(float(record["r"]) - aspect) > stored_tol:
            add_error(errors, "STORED_ASPECT_MISMATCH", index=index)
        if L < W:
            long_axis_reversed += 1

        expected_ng = int(spec["nG_requested"])
        if int(record["nG_actual"]) != expected_ng or int(record["retry_nG"]) != expected_ng:
            add_error(errors, "UNEXPECTED_NG_OR_RETRY", index=index)
        if bool(record["isolated"]):
            add_error(errors, "ISOLATED_RECORD_PRESENT", index=index)
        if record["material"] != spec["material"] or record["substrate"] != spec["substrate"]:
            add_error(errors, "RECORD_MATERIAL_MISMATCH", index=index)

        key = (L, W, H, P, pol)
        geometry = (L, W, H, P)
        keys.append(key)
        geometry_pols.setdefault(geometry, set()).add(pol)

    if len(records) != expected_records:
        add_error(errors, "RECORD_COUNT_MISMATCH", expected=expected_records, actual=len(records))
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        add_error(errors, "DUPLICATE_KEYS", count=duplicate_count)
    partial = [geometry for geometry, pols in geometry_pols.items() if pols != polarizations]
    if partial:
        add_error(errors, "INCOMPLETE_POLARIZATION_PAIRS", count=len(partial))
    expected_geometries = expected_records // len(polarizations)
    if len(geometry_pols) != expected_geometries:
        add_error(
            errors,
            "GEOMETRY_COUNT_MISMATCH",
            expected=expected_geometries,
            actual=len(geometry_pols),
        )

    for key, expected in spec["expected_meta"].items():
        actual = meta.get(key, "<missing>")
        if actual != expected:
            add_error(errors, "META_MISMATCH", field=key, expected=expected, actual=actual)

    if long_axis_reversed:
        warnings.append(
            {
                "code": "AXIS_CANONICALIZATION_REQUIRED",
                "records": long_axis_reversed,
                "instruction": "Canonicalize axes and swap p/s channels when axes swap.",
            }
        )
    warnings.append(
        {
            "code": "NG_FIELD_SEMANTICS",
            "instruction": "Record requested=131 and retained=121 separately in derived manifests.",
        }
    )

    fatal_codes = {item["code"] for item in errors}
    checkpoint_only = {
        "RECORD_COUNT_MISMATCH",
        "GEOMETRY_COUNT_MISMATCH",
        "INCOMPLETE_POLARIZATION_PAIRS",
    }
    healthy_checkpoint = not (fatal_codes - checkpoint_only)
    completed = len(records) == expected_records
    rt_array = np.asarray(rt_means, dtype=float)
    display_extrema = {
        key: (None if math.isinf(value) else value) for key, value in extrema.items()
    }
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "passed": completed and not errors,
        "healthy_checkpoint": healthy_checkpoint,
        "records": len(records),
        "expected_records": expected_records,
        "success_records": len(keys),
        "geometries": len(geometry_pols),
        "complete_pairs": len(geometry_pols) - len(partial),
        "partial_pairs": len(partial),
        "duplicate_keys": duplicate_count,
        "R_plus_T_mean": float(np.mean(rt_array)) if rt_array.size else None,
        "R_plus_T_min": float(np.min(rt_array)) if rt_array.size else None,
        "R_plus_T_max": float(np.max(rt_array)) if rt_array.size else None,
        "pointwise_conservation_error_max": pointwise_error_max,
        **display_extrema,
        "meta": meta,
        "errors": errors[:100],
        "error_count": len(errors),
        "warnings": warnings,
        "sha256": file_digest(path),
        "md5": file_digest(path, "md5"),
        "size_bytes": path.stat().st_size,
    }


def audit_protected_files(policy: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    assets = [
        *policy.get("protected_files", []),
        *policy.get("immutable_assets", []),
    ]
    for item in assets:
        path = ROOT / item["path"]
        actual = file_digest(path, "md5") if path.exists() else None
        results.append(
            {
                "path": item["path"],
                "expected_md5": item["md5"],
                "actual_md5": actual,
                "passed": actual == item["md5"],
            }
        )
    return results


def verify_policy_integrity(policy: dict[str, Any]) -> dict[str, Any]:
    """Verify the policy and supervisor code against a separately pinned lock."""
    spec = policy.get("integrity", {})
    if not isinstance(spec, dict) or spec.get("enforce") is not True:
        return {"enforced": False, "passed": True}
    lock_value = spec.get("lock_path")
    lock_path = workspace_file(lock_value)
    if lock_path is None or not lock_path.is_file():
        return {
            "enforced": True,
            "passed": False,
            "error": "integrity lock is missing or outside the workspace",
        }
    try:
        lock = load_json(lock_path, {}) or {}
        expected_policy = str(lock.get("policy_sha256", "")).upper()
        expected_supervisor = str(lock.get("supervisor_sha256", "")).upper()
        actual_policy = file_digest(POLICY).upper()
        actual_supervisor = file_digest(ROOT / "pipeline_supervisor.py").upper()
    except Exception as exc:
        return {
            "enforced": True,
            "passed": False,
            "error": f"integrity lock unreadable: {type(exc).__name__}: {exc}",
        }
    passed = bool(expected_policy and expected_supervisor)
    passed = passed and actual_policy == expected_policy and actual_supervisor == expected_supervisor
    return {
        "enforced": True,
        "passed": passed,
        "lock_path": str(lock_path.relative_to(ROOT)).replace("\\", "/"),
        "expected_policy_sha256": expected_policy or None,
        "actual_policy_sha256": actual_policy,
        "expected_supervisor_sha256": expected_supervisor or None,
        "actual_supervisor_sha256": actual_supervisor,
    }


def recovery_attempt(status: dict[str, Any]) -> int:
    recovery = status.get("recovery")
    if not isinstance(recovery, dict):
        return 0
    return int(recovery.get("attempt", 1))


def verify_file_binding(binding: Any, label: str) -> tuple[bool, str | None]:
    if not isinstance(binding, dict):
        return False, f"{label} binding is missing"
    path = workspace_file(binding.get("path"))
    expected = str(binding.get("sha256", "")).upper()
    if path is None or not path.is_file():
        return False, f"{label} file is missing or outside workspace"
    if not expected or file_digest(path) != expected:
        return False, f"{label} SHA256 mismatch"
    return True, None


def verify_d65_gate(payload: dict[str, Any], pool: dict[str, Any]) -> tuple[bool, str | None]:
    required_checks = {
        "pool_records_6000",
        "pool_grid_exact",
        "perfect_reflector_lab_neutral",
        "perfect_reflector_d65_xy",
        "black_reflector_lab_zero",
        "lab_source_unclipped_xyz",
        "srgb_display_only",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or any(checks.get(name) is not True for name in required_checks):
        return False, "D65 evidence lacks the complete passed check set"
    if int(payload.get("evidence_revision", 0)) < 2:
        return False, "D65 evidence revision must be at least 2"
    source_pool = payload.get("pool")
    if not isinstance(source_pool, dict):
        return False, "D65 evidence lacks its source pool"
    source_path = workspace_file(source_pool.get("path"))
    source_sha = str(source_pool.get("sha256", "")).upper()
    if source_path is None or not source_path.is_file() or file_digest(source_path) != source_sha:
        return False, "D65 source pool SHA256 mismatch"
    if int(source_pool.get("records", -1)) <= 0:
        return False, "D65 source pool record count is invalid"
    for field, label in (("implementation", "D65 implementation"), ("tests", "D65 tests")):
        valid, error = verify_file_binding(payload.get(field), label)
        if not valid:
            return False, error
    provenance = payload.get("derived_label_provenance", {})
    if (
        provenance.get("lab_source") != "direct_unclipped_xyz"
        or provenance.get("srgb_role") != "display_only_clipped"
    ):
        return False, "D65 label provenance is invalid"
    references = payload.get("reference_cases", {})
    try:
        white_lab = np.asarray(references["perfect_reflector"]["lab"], dtype=float)
        black_lab = np.asarray(references["black_reflector"]["lab"], dtype=float)
        white_xy = np.asarray(references["white_xy"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False, "D65 reference cases are malformed"
    if white_lab.shape != (3,) or np.max(np.abs(white_lab - [100.0, 0.0, 0.0])) > 1e-10:
        return False, "D65 perfect-reflector Lab check is invalid"
    if black_lab.shape != (3,) or np.max(np.abs(black_lab)) > 1e-12:
        return False, "D65 black-reflector Lab check is invalid"
    if white_xy.shape != (2,) or np.max(np.abs(white_xy - [0.3127, 0.3290])) > 5e-4:
        return False, "D65 white-point check is invalid"
    if payload.get("legacy_path_modified") is not False:
        return False, "D65 evidence does not preserve the legacy path"
    return True, None


def all_checks_true(
    checks: Any, required: set[str] | None = None
) -> tuple[bool, str | None]:
    if not isinstance(checks, dict) or not checks:
        return False, "checks must be a non-empty object"
    if required is not None and set(checks) != required:
        return False, "check set differs from the registered contract"
    failed = [name for name, value in checks.items() if value is not True]
    if failed:
        return False, f"checks are not all true: {failed}"
    return True, None


def bindings_exist(bindings: Any, label: str) -> tuple[bool, str | None]:
    if isinstance(bindings, dict) and {"path", "sha256"} <= set(bindings):
        bindings = [bindings]
    if not isinstance(bindings, list) or not bindings:
        return False, f"{label} bindings are missing"
    for index, binding in enumerate(bindings):
        valid, error = verify_file_binding(binding, f"{label}[{index}]")
        if not valid:
            return False, error
    return True, None


def runtime_hashes_match(
    runtime_hashes: Any, required: set[str] | None = None
) -> tuple[bool, str | None]:
    if not isinstance(runtime_hashes, dict) or not runtime_hashes:
        return False, "runtime hashes are missing"
    if required is not None and set(runtime_hashes) != required:
        return False, "runtime hash set differs from the registered contract"
    for name, expected in runtime_hashes.items():
        path = workspace_file(name)
        if path is None or not path.is_file():
            return False, f"runtime path is missing or outside workspace: {name}"
        if file_digest(path) != str(expected).upper():
            return False, f"runtime SHA256 mismatch: {name}"
    return True, None


def bound_json(binding: Any, label: str) -> tuple[dict[str, Any] | None, str | None]:
    valid, error = verify_file_binding(binding, label)
    if not valid:
        return None, error
    path = workspace_file(binding["path"])
    try:
        payload = load_json(path, {}) or {}
    except Exception as exc:
        return None, f"{label} is not valid JSON: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{label} must contain a JSON object"
    return payload, None


def valid_request_identity(request: Any) -> bool:
    return bool(
        isinstance(request, dict)
        and set(request) == {"request_id", "attempt"}
        and isinstance(request.get("request_id"), str)
        and request["request_id"]
        and isinstance(request.get("attempt"), int)
        and request["attempt"] >= 1
    )


def current_request_identity(action: str) -> dict[str, Any]:
    dispatch = load_json(DISPATCH_REQUEST, {}) or {}
    if (
        dispatch.get("action") != action
        or dispatch.get("status") not in {"pending", "in_progress", "failed"}
        or not valid_request_identity({
            "request_id": dispatch.get("request_id"),
            "attempt": dispatch.get("attempt"),
        })
    ):
        raise ValueError(f"active dispatch identity is invalid for {action}")
    return {
        "request_id": dispatch["request_id"],
        "attempt": int(dispatch["attempt"]),
    }


def reusable_evidence_request(request: Any, action: str) -> dict[str, Any]:
    if not valid_request_identity(request):
        raise ValueError(f"stored evidence request identity is invalid for {action}")
    active = current_request_identity(action)
    if (
        request["request_id"] != active["request_id"]
        or int(request["attempt"]) > int(active["attempt"])
    ):
        raise ValueError(f"stored evidence request is not reusable for {action}")
    return {"request_id": request["request_id"], "attempt": int(request["attempt"])}


def request_identity_authorized(request: Any, action: str) -> bool:
    if not valid_request_identity(request):
        return False
    dispatch = load_json(DISPATCH_REQUEST, {}) or {}
    if (
        dispatch.get("action") == action
        and dispatch.get("request_id") == request["request_id"]
        and int(dispatch.get("attempt", 0)) >= int(request["attempt"])
    ):
        return True
    history = STATE / "dispatch_history" / (
        f"{request['request_id']}-attempt{int(request['attempt'])}.json"
    )
    archived = load_json(history, {}) or {}
    archived_request = archived.get("request", {})
    return bool(
        archived_request.get("action") == action
        and archived_request.get("request_id") == request["request_id"]
        and int(archived_request.get("attempt", 0)) == int(request["attempt"])
    )


def failed_ack_authorizes_worker(
    worker_binding: Any, request: Any, action: str
) -> bool:
    """Bind diagnostic activation to the exact evidence named by its terminal ack."""
    if not valid_request_identity(request):
        return False

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    dispatch = load_json(DISPATCH_REQUEST, {}) or {}
    ack = load_json(EXECUTOR_ACK, {}) or {}
    if (
        dispatch.get("action") == action
        and dispatch.get("request_id") == request["request_id"]
        and int(dispatch.get("attempt", 0)) == int(request["attempt"])
    ):
        candidates.append((dispatch, ack))
    history_path = STATE / "dispatch_history" / (
        f"{request['request_id']}-attempt{int(request['attempt'])}.json"
    )
    archived = load_json(history_path, {}) or {}
    if isinstance(archived.get("request"), dict):
        candidates.append((archived["request"], archived.get("final_ack") or {}))

    for bound_request, bound_ack in candidates:
        if (
            bound_request.get("action") == action
            and bound_request.get("status") == "failed"
            and bound_request.get("terminal_failure") is True
            and str(bound_request.get("failure_class", "")).lower() == "scientific"
            and bound_ack.get("request_id") == request["request_id"]
            and int(bound_ack.get("attempt", 0)) == int(request["attempt"])
            and bound_ack.get("status") == "failed"
            and str(bound_ack.get("failure_class", "")).lower() == "scientific"
            and worker_binding in bound_ack.get("evidence", [])
        ):
            return True
    return False


def audited_worker_payload(
    payload: dict[str, Any],
    *,
    action: str,
    audit_version: str,
    worker_version: str,
    audit_fields: set[str],
    auditor_runtime_paths: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate an auditor envelope without executing worker-controlled artifacts."""
    if (
        payload.get("schema_version") != 1
        or payload.get("evidence_version") != audit_version
        or payload.get("independent_reproduction") is not True
        or not request_identity_authorized(payload.get("request"), action)
    ):
        return None, f"{action} independent audit identity is invalid"
    worker, error = bound_json(payload.get("worker_evidence"), f"{action} worker evidence")
    if error:
        return None, error
    worker_request = worker.get("request")
    audit_request = payload.get("request")
    if (
        worker.get("schema_version") != 1
        or worker.get("evidence_version") != worker_version
        or not valid_request_identity(worker_request)
        or worker_request["request_id"] != audit_request["request_id"]
        or int(worker_request["attempt"]) > int(audit_request["attempt"])
    ):
        return None, f"{action} worker evidence identity is invalid"
    valid, error = runtime_hashes_match(
        payload.get("auditor_runtime_hashes"), auditor_runtime_paths
    )
    if not valid:
        return None, error
    expected = copy.deepcopy(worker)
    expected["evidence_version"] = audit_version
    expected["request"] = audit_request
    expected["worker_evidence"] = payload.get("worker_evidence")
    expected["independent_reproduction"] = True
    expected["auditor_runtime_hashes"] = payload.get("auditor_runtime_hashes")
    for name in audit_fields:
        expected[name] = payload.get(name)
    if expected != payload:
        return None, f"{action} audit is not an exact extension of its worker evidence"
    return worker, None


def verify_protected_snapshot(snapshot: Any) -> tuple[bool, str | None]:
    if not isinstance(snapshot, list) or not snapshot:
        return False, "protected-file snapshot is missing"
    policy = load_policy(POLICY)
    required_paths = {
        item["path"]
        for item in (*policy.get("protected_files", []), *policy.get("immutable_assets", []))
    }
    snapshot_paths = {
        item.get("path") for item in snapshot if isinstance(item, dict)
    }
    if snapshot_paths != required_paths:
        return False, "protected-file snapshot differs from the complete policy set"
    seen = set()
    for item in snapshot:
        if not isinstance(item, dict) or item.get("passed") is not True:
            return False, "protected-file snapshot contains a failed item"
        name = item.get("path")
        path = workspace_file(name)
        expected = str(item.get("expected_md5", "")).upper()
        actual = str(item.get("actual_md5", "")).upper()
        if path is None or not path.is_file() or not expected or expected != actual:
            return False, f"protected-file binding is invalid: {name}"
        if name in seen or file_digest(path, "md5") != expected:
            return False, f"protected-file digest mismatch or duplicate: {name}"
        seen.add(name)
    return True, None


def finite_values(values: Any, count: int) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (count,) or not np.isfinite(array).all():
        return None
    return array


REFERENCE_COMPARISON_NAMES = {
    "order_365x768_to_450x768_0p5nm",
    "grid_450x512_to_450x768_0p5nm",
    "corner_365x512_to_450x768_0p5nm",
    "spectral_450x768_1nm_to_0p5nm",
    "frozen_candidate_to_final_reference",
}


def verify_reference_comparisons(
    comparisons: Any, count: int, require_pass: bool, *, start_index: int
) -> tuple[bool, str | None]:
    if not isinstance(comparisons, dict) or set(comparisons) != REFERENCE_COMPARISON_NAMES:
        return False, "reference comparison set differs from the registered contract"
    for name, comparison in comparisons.items():
        if not isinstance(comparison, dict) or int(comparison.get("count", -1)) != count:
            return False, f"reference comparison count is invalid: {name}"
        values = finite_values(comparison.get("joint_max_by_geometry"), count)
        rows = comparison.get("rows")
        if values is None or np.any(values < 0) or not isinstance(rows, list) or len(rows) != count * 2:
            return False, f"reference comparison raw values are invalid: {name}"
        row_values: dict[tuple[int, str], float] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"geometry_index", "pol", "dE00"}:
                return False, f"reference comparison row schema is invalid: {name}"
            geometry_index = row.get("geometry_index")
            pol = row.get("pol")
            try:
                value = float(row.get("dE00"))
            except (TypeError, ValueError):
                return False, f"reference comparison row value is invalid: {name}"
            key = (geometry_index, pol)
            if (
                not isinstance(geometry_index, int)
                or isinstance(geometry_index, bool)
                or geometry_index not in range(start_index, start_index + count)
                or pol not in {"p", "s"}
                or key in row_values
                or not math.isfinite(value)
                or value < 0
            ):
                return False, f"reference comparison row identity is invalid: {name}"
            row_values[key] = value
        row_joint = np.asarray(
            [
                max(row_values[(index, "p")], row_values[(index, "s")])
                for index in range(start_index, start_index + count)
            ],
            dtype=float,
        )
        if not np.allclose(values, row_joint, rtol=0.0, atol=1e-12):
            return False, f"reference comparison rows differ from joint values: {name}"
        mean = float(np.mean(values))
        maximum = float(np.max(values))
        if (
            not math.isclose(float(comparison.get("mean", float("nan"))), mean, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(comparison.get("max", float("nan"))), maximum, rel_tol=0.0, abs_tol=1e-12)
        ):
            return False, f"reference comparison statistics are inconsistent: {name}"
        mean_passed = mean < 1.15
        all_passed = maximum < 2.3
        if (
            comparison.get("mean_lt_1_15") is not mean_passed
            or comparison.get("all_lt_2_3") is not all_passed
            or comparison.get("passed") is not (mean_passed and all_passed)
            or (require_pass and comparison.get("passed") is not True)
        ):
            return False, f"reference comparison verdict is inconsistent: {name}"
    return True, None


def verify_active_protocol_bindings(
    active_binding: Any, protocol_binding: Any, pool_sha256: str
) -> tuple[bool, str | None]:
    active, error = bound_json(active_binding, "active pool")
    if error:
        return False, error
    protocol, error = bound_json(protocol_binding, "approved protocol")
    if error:
        return False, error
    if (
        active.get("schema_version") != 1
        or active.get("evidence_version") != "paper2-active-pool-v1"
        or active.get("active") is not True
        or active.get("training_allowed") is not False
        or str(active.get("pool_sha256", "")).upper() != pool_sha256
        or active.get("approved_protocol") != protocol_binding
    ):
        return False, "active-pool binding is invalid"
    if (
        protocol.get("schema_version") != 1
        or protocol.get("evidence_version") != "paper2-replacement-protocol-v1"
        or protocol.get("protocol_revision") != "v2_bound_holdout"
        or protocol.get("approved") is not True
        or protocol.get("automatic_launch_authorized") is not True
        or active.get("pool_spec") != protocol.get("pool_spec")
    ):
        return False, "approved protocol binding is invalid"
    return True, None


def verify_reference_resolution_gate(
    payload: dict[str, Any], _pool: dict[str, Any]
) -> tuple[bool, str | None]:
    if payload.get("schema_version") != 1 or not production_reference_audit_approved(payload):
        return False, "reference audit is not the registered v2 holdout pass"
    if not request_identity_authorized(payload.get("request"), "reference_resolution"):
        return False, "reference audit request identity is invalid"
    if payload.get("training_allowed") is not False:
        return False, "reference evidence must keep training disabled"
    if payload.get("independent_reproduction") is not True:
        return False, "reference audit is not an independent reproduction"
    valid, error = runtime_hashes_match(
        payload.get("auditor_runtime_hashes"),
        AUDITOR_RUNTIME_PATHS["reference_resolution"],
    )
    if not valid:
        return False, error
    if payload.get("primary_gate_population") != "24_new_holdout_geometries_only":
        return False, "reference gate population is not the frozen 24-case holdout"
    if payload.get("combined_32_population_scope") != "supplemental_reporting_only":
        return False, "combined 32-case population is not supplemental only"
    valid, error = all_checks_true(
        payload.get("checks"),
        {
            "policy_integrity",
            "paper1_and_legacy_assets_unchanged",
            "v2_plan_and_all_source_hashes_verified",
            "candidate_independently_refrozen_on_initial_eight",
            "holdout_did_not_reselect_candidate",
            "exact_extension_task_set",
            "worker_evidence_exactly_reproduced",
            "production_reference_approved",
        },
    )
    if not valid:
        return False, error
    if payload.get("thresholds") != {
        "mean_joint_dE00_lt": 1.15,
        "all_joint_dE00_lt": 2.3,
        "pointwise_conservation_lte": 1e-6,
    }:
        return False, "reference thresholds differ from the registered contract"
    candidate = payload.get("approved_protocol_candidate")
    if not isinstance(candidate, dict) or candidate.get("passed") is not True:
        return False, "reference audit lacks the frozen passed candidate"
    for field in ("requested_nG", "Nxy", "wavelength_step_nm"):
        if field not in candidate:
            return False, f"reference candidate lacks {field}"
    sources = payload.get("sources")
    required_sources = {
        "plan",
        "source_v2_plan",
        "source_v2_checkpoint",
        "source_v2_worker_evidence",
        "source_v2_independent_audit",
        "source_base_checkpoint",
        "holdout_evidence",
        "holdout_checkpoint",
    }
    if not isinstance(sources, dict) or set(sources) != required_sources:
        return False, "reference source binding set differs from the registered contract"
    if payload.get("worker_evidence") != sources.get("holdout_evidence"):
        return False, "reference audit worker binding differs from its source ledger"
    valid, error = bindings_exist(list(sources.values()), "reference source")
    if not valid:
        return False, error
    plan, error = bound_json(sources["plan"], "reference holdout plan")
    if error:
        return False, error
    if (
        plan.get("final_reference") != REFERENCE_HOLDOUT_FINAL_REFERENCE
        or plan.get("frozen_candidate") != candidate
        or plan.get("thresholds") != payload.get("thresholds")
        or str(plan.get("pool", {}).get("sha256", "")).upper()
        != str(payload.get("pool_sha256", "")).upper()
        or plan.get("primary_gate_population") != payload.get("primary_gate_population")
        or plan.get("combined_32_population_scope")
        != payload.get("combined_32_population_scope")
    ):
        return False, "reference audit differs from its frozen holdout plan"
    valid, error = verify_file_binding(plan.get("pool"), "reference holdout pool")
    if not valid:
        return False, error
    for name in (
        "source_v2_plan",
        "source_v2_checkpoint",
        "source_v2_worker_evidence",
        "source_v2_independent_audit",
        "source_base_checkpoint",
    ):
        if sources[name] != plan.get(name):
            return False, f"reference source differs from the frozen plan: {name}"
    valid, error = verify_reference_comparisons(
        payload.get("independent_holdout_comparisons"), 24, True, start_index=8
    )
    if not valid:
        return False, error
    valid, error = verify_reference_comparisons(
        payload.get("combined_32_supplemental_comparisons"), 32, False, start_index=0
    )
    if not valid:
        return False, error
    return verify_protected_snapshot(payload.get("protected_files"))


def verify_replacement_pool_gate(
    payload: dict[str, Any], pool: dict[str, Any]
) -> tuple[bool, str | None]:
    worker, error = audited_worker_payload(
        payload,
        action="replacement_pool_generation",
        audit_version="paper2-replacement-pool-audit-v1",
        worker_version="paper2-replacement-pool-v1",
        audit_fields={"independent_audit"},
        auditor_runtime_paths=AUDITOR_RUNTIME_PATHS["replacement_pool_generation"],
    )
    if error:
        return False, error
    assert worker is not None
    if not failed_ack_authorizes_worker(
        payload.get("worker_evidence"), payload.get("request"),
        "replacement_pool_generation",
    ):
        return False, "replacement worker evidence is not authorized by the failed ack"
    if payload.get("schema_version") != 1 or payload.get("training_allowed") is not False:
        return False, "replacement evidence schema or training lock is invalid"
    pool_sha = str(payload.get("pool_sha256", "")).upper()
    if not pool_sha or pool_sha != str(pool.get("sha256", "")).upper():
        return False, "replacement evidence does not match the active pool"
    spec = payload.get("pool_spec")
    if not isinstance(spec, dict) or int(spec.get("expected_records", 0)) <= 0:
        return False, "replacement pool_spec is invalid"
    pool_path = workspace_file(spec.get("path"))
    if pool_path is None or not pool_path.is_file() or file_digest(pool_path) != pool_sha:
        return False, "replacement pool file or SHA256 is invalid"
    approved = payload.get("approved_protocol")
    protocol, error = bound_json(approved, "approved protocol")
    if error:
        return False, error
    if (
        protocol.get("schema_version") != 1
        or protocol.get("evidence_version") != "paper2-replacement-protocol-v1"
        or protocol.get("protocol_revision") != "v2_bound_holdout"
        or protocol.get("approved") is not True
        or protocol.get("automatic_launch_authorized") is not True
        or protocol.get("pool_spec") != spec
    ):
        return False, "approved replacement protocol is invalid"
    reference = payload.get("reference_gate_evidence")
    if reference != protocol.get("source_reference_gate"):
        return False, "replacement reference binding differs from its protocol"
    reference_payload, error = bound_json(reference, "replacement reference gate")
    if error or not production_reference_audit_approved(reference_payload):
        return False, error or "replacement reference gate is not approved"
    activation_id = hashlib.sha256(
        f"{str(approved['sha256']).upper()}|{pool_sha}".encode("ascii")
    ).hexdigest()[:24]
    if payload.get("activation_id") != activation_id:
        return False, "replacement activation_id is not hash-bound"
    audit = payload.get("audit")
    audit_fields = {
        "records", "expected_records", "geometries", "complete_pairs",
        "duplicate_keys", "R_plus_T_mean", "R_plus_T_min", "R_plus_T_max",
        "pointwise_conservation_error_max", "R_min", "R_max", "T_min", "T_max",
    }
    if not isinstance(audit, dict) or set(audit) != audit_fields:
        return False, "replacement strict audit is missing"
    if payload.get("independent_audit") != audit:
        return False, "replacement independent audit differs from the worker audit"
    if pool.get("passed") is not True or any(audit.get(name) != pool.get(name) for name in audit_fields):
        return False, "replacement strict audit metrics are invalid"
    if (
        payload.get("pool_md5") != file_digest(pool_path, "md5")
        or payload.get("size_bytes") != pool_path.stat().st_size
    ):
        return False, "replacement pool size or MD5 mismatch"
    checkpoint = payload.get("checkpoint")
    valid, error = bindings_exist(checkpoint, "replacement checkpoint")
    if not valid:
        return False, error
    if not isinstance(checkpoint.get("failure_events"), int) or checkpoint["failure_events"] < 0:
        return False, "replacement checkpoint failure ledger is invalid"
    valid, error = runtime_hashes_match(
        payload.get("runtime_hashes"),
        {"rcwa_batch.py", "paper2_colorimetry_fine.py", "scripts/run_replacement_pool.py"},
    )
    if not valid:
        return False, error
    return verify_protected_snapshot(payload.get("protected_files"))


def verify_joint_v2_gate(
    payload: dict[str, Any], pool: dict[str, Any]
) -> tuple[bool, str | None]:
    worker, error = audited_worker_payload(
        payload,
        action="joint_numerical_convergence",
        audit_version="paper2-joint-convergence-audit-v1",
        worker_version="paper2-joint-convergence-v2",
        audit_fields={"independent_evaluation"},
        auditor_runtime_paths=AUDITOR_RUNTIME_PATHS["joint_numerical_convergence"],
    )
    if error:
        return False, error
    assert worker is not None
    if payload.get("classification") != "passed" or payload.get("training_allowed") is not False:
        return False, "joint-v2 classification or training lock is invalid"
    if str(payload.get("pool_sha256", "")).upper() != str(pool.get("sha256", "")).upper():
        return False, "joint-v2 pool SHA256 mismatch"
    valid, error = all_checks_true(
        payload.get("checks"),
        {
            "active_pool_strict_audit",
            "paper1_and_legacy_assets_unchanged",
            "replacement_vs_reference",
        },
    )
    if not valid:
        return False, error
    if payload.get("thresholds") != {
        "mean_joint_dE00_lt": 1.15,
        "all_joint_dE00_lt": 2.3,
        "pointwise_conservation_lte": 1e-6,
        "stored_label_atol": 1e-10,
    }:
        return False, "joint-v2 thresholds differ from the registered contract"
    evaluation = payload.get("evaluation", {})
    if payload.get("independent_evaluation") != evaluation:
        return False, "joint-v2 independent evaluation differs from worker evidence"
    valid, error = all_checks_true(
        evaluation.get("checks"),
        {
            "exact_32_complete_p_s_geometries",
            "derived_labels_exact",
            "pool_conservation",
            "reference_conservation",
            "mean_joint_dE00",
            "all_joint_dE00",
        },
    )
    if not valid or evaluation.get("passed") is not True:
        return False, error or "joint-v2 independent evaluation failed"
    joint = evaluation.get("joint_dE00", {})
    values = finite_values(joint.get("values"), 32)
    if int(joint.get("count", -1)) != 32 or values is None or np.any(values < 0):
        return False, "joint-v2 does not contain the frozen 32 geometries"
    expected_stats = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }
    if any(
        not math.isclose(float(joint.get(name, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12)
        for name, expected in expected_stats.items()
    ):
        return False, "joint-v2 summary statistics do not match raw values"
    if expected_stats["mean"] >= 1.15 or expected_stats["max"] >= 2.3:
        return False, "joint-v2 raw values fail the registered thresholds"
    if (
        float(evaluation.get("pointwise_conservation_error_max", float("inf"))) > 1e-6
        or float(evaluation.get("reference_pointwise_conservation_error_max", float("inf"))) > 1e-6
    ):
        return False, "joint-v2 raw conservation metrics fail"
    if evaluation.get("missing") or evaluation.get("label_failures"):
        return False, "joint-v2 contains missing spectra or label failures"
    for name in ("active_pool", "approved_protocol", "reference_gate"):
        valid, error = bindings_exist(payload.get(name), name)
        if not valid:
            return False, error
    valid, error = verify_active_protocol_bindings(
        payload.get("active_pool"), payload.get("approved_protocol"),
        str(payload.get("pool_sha256", "")).upper(),
    )
    if not valid:
        return False, error
    valid, error = runtime_hashes_match(
        payload.get("runtime_hashes"),
        {"paper2_colorimetry_fine.py", "scripts/run_joint_convergence_v2.py"},
    )
    if not valid:
        return False, error
    reference, error = bound_json(payload.get("reference_gate"), "joint-v2 reference gate")
    if error or not production_reference_audit_approved(reference):
        return False, error or "joint-v2 reference gate is not approved"
    raw = payload.get("reference_raw_spectra")
    sources = reference.get("sources", {})
    expected_raw = {
        "base_checkpoint_sha256": str(sources.get("source_base_checkpoint", {}).get("sha256", "")).upper(),
        "budget_v2_checkpoint_sha256": str(sources.get("source_v2_checkpoint", {}).get("sha256", "")).upper(),
        "holdout_checkpoint_sha256": str(sources.get("holdout_checkpoint", {}).get("sha256", "")).upper(),
    }
    if raw != expected_raw:
        return False, "joint-v2 raw reference hashes differ from the approved gate"
    return verify_protected_snapshot(payload.get("protected_files"))


def verify_cross_solver_v2_gate(
    payload: dict[str, Any], pool: dict[str, Any]
) -> tuple[bool, str | None]:
    worker, error = audited_worker_payload(
        payload,
        action="cross_solver_spectrum_validation",
        audit_version="paper2-cross-solver-audit-v1",
        worker_version="paper2-cross-solver-v2",
        audit_fields={"independent_controls", "independent_evaluation"},
        auditor_runtime_paths=AUDITOR_RUNTIME_PATHS[
            "cross_solver_spectrum_validation"
        ],
    )
    if error:
        return False, error
    assert worker is not None
    if payload.get("classification") != "passed" or payload.get("training_allowed") is not False:
        return False, "cross-solver classification or training lock is invalid"
    if str(payload.get("pool_sha256", "")).upper() != str(pool.get("sha256", "")).upper():
        return False, "cross-solver pool SHA256 mismatch"
    valid, error = all_checks_true(
        payload.get("checks"),
        {
            "controls",
            "matched_results",
            "runtime_hashes_verified",
            "paper1_and_legacy_assets_unchanged",
        },
    )
    if not valid:
        return False, error
    controls = payload.get("controls", {})
    if payload.get("independent_controls") != controls:
        return False, "cross-solver independent controls differ from worker evidence"
    control_names = {
        f"{solver}_{metric}"
        for solver in ("grcwa", "thirdparty")
        for metric in (
            "fresnel_error", "empty_energy_error", "circle_max_difference",
            "rotation_max_difference",
        )
    }
    valid, error = all_checks_true(controls.get("checks"), control_names)
    if not valid or controls.get("passed") is not True:
        return False, error or "cross-solver controls failed"
    evaluation = payload.get("evaluation", {})
    if payload.get("independent_evaluation") != evaluation:
        return False, "cross-solver independent evaluation differs from worker evidence"
    valid, error = all_checks_true(
        evaluation.get("checks"),
        {"no_task_failures", "production_cross_solver", "all_stress_cross_solver", "both_solvers_converged"},
    )
    if not valid or evaluation.get("passed") is not True:
        return False, error or "cross-solver evaluation failed"
    protocol = payload.get("protocol", {})
    production = protocol.get("production")
    stress = protocol.get("stress_configs")
    selected = payload.get("selected_geometries")
    if (
        int(protocol.get("geometry_count", -1)) != 12
        or int(protocol.get("stress_geometry_count", -1)) != 4
        or protocol.get("polarizations") != ["p", "s"]
        or protocol.get("background") != "air"
        or protocol.get("incident") != "air"
        or protocol.get("transmission_halfspace") != "SiO2"
        or protocol.get("wavelength_nm") != np.arange(380.0, 785.0, 5.0).tolist()
        or not isinstance(production, dict)
        or set(production) != {"nG_requested", "nG_retained", "Nxy"}
        or int(production.get("nG_requested", 0)) < 1
        or int(production.get("nG_retained", 0)) < 1
        or int(round(math.sqrt(int(production.get("nG_retained", 0))))) ** 2
        != int(production.get("nG_retained", 0))
        or int(round(math.sqrt(int(production.get("nG_retained", 0))))) % 2 != 1
        or int(production.get("Nxy", 0)) < 64
        or not isinstance(stress, list)
        or not stress
        or not isinstance(selected, list)
        or len(selected) != 12
        or sum(item.get("stress") is True for item in selected if isinstance(item, dict)) != 4
    ):
        return False, "cross-solver protocol differs from the registered design"
    production_nG = int(production["nG_requested"])
    production_Nxy = int(production["Nxy"])
    expected_stress = []
    if production_nG < 365:
        expected_stress.append({"name": "order_axis", "nG_requested": 365, "Nxy": production_Nxy})
    if production_Nxy < 512:
        expected_stress.append({"name": "grid_axis", "nG_requested": production_nG, "Nxy": 512})
    if production_nG < 365 or production_Nxy < 512:
        expected_stress.append({"name": "reference_corner", "nG_requested": 365, "Nxy": 512})
    else:
        expected_stress.extend([
            {"name": "order_axis", "nG_requested": 450, "Nxy": production_Nxy},
            {"name": "grid_axis", "nG_requested": production_nG, "Nxy": 768},
            {"name": "higher_corner", "nG_requested": 450, "Nxy": 768},
        ])
    unique_stress = []
    seen_configs = set()
    for config in expected_stress:
        key = (config["nG_requested"], config["Nxy"])
        if key not in seen_configs and key != (production_nG, production_Nxy):
            unique_stress.append(config)
            seen_configs.add(key)
    if stress != unique_stress:
        return False, "cross-solver stress configuration differs from the registered design"
    stress_names = []
    for config in stress:
        if (
            not isinstance(config, dict)
            or set(config) != {"name", "nG_requested", "Nxy"}
            or not isinstance(config.get("name"), str)
            or not config["name"]
            or config["name"] == "production"
            or int(config.get("nG_requested", 0)) < int(production["nG_requested"])
            or int(config.get("Nxy", 0)) < int(production["Nxy"])
        ):
            return False, "cross-solver stress configuration is invalid"
        stress_names.append(config["name"])
    if len(stress_names) != len(set(stress_names)):
        return False, "cross-solver stress configuration names are duplicated"
    if payload.get("thresholds") != {
        "per_spectrum_R_T_rmse_lte": 0.05,
        "mean_spectrum_R_T_rmse_lte": 0.03,
        "mean_joint_dE00_lt": 1.15,
        "per_geometry_joint_dE00_lt": 2.3,
        "energy_error_lte": 1e-6,
        "analytic_and_symmetry_error_lte": 1e-7,
    }:
        return False, "cross-solver thresholds differ from the registered contract"
    for name in ("raw_checkpoint", "active_pool", "approved_protocol"):
        valid, error = bindings_exist(payload.get(name), name)
        if not valid:
            return False, error
    valid, error = verify_active_protocol_bindings(
        payload.get("active_pool"), payload.get("approved_protocol"),
        str(payload.get("pool_sha256", "")).upper(),
    )
    if not valid:
        return False, error
    valid, error = runtime_hashes_match(
        payload.get("runtime_hashes"),
        {
            "rcwa_batch.py",
            "paper2_colorimetry_fine.py",
            "scripts/run_cross_solver_validation.py",
            "scripts/run_cross_solver_validation_v2.py",
        },
    )
    if not valid:
        return False, error
    if evaluation.get("classification") != "passed" or evaluation.get("failures"):
        return False, "cross-solver evaluation classification is invalid"
    return verify_protected_snapshot(payload.get("protected_files"))


GATE_PAYLOAD_VERIFIERS = {
    "d65_colorimetry": verify_d65_gate,
    "reference_resolution": verify_reference_resolution_gate,
    "replacement_pool_ready": verify_replacement_pool_gate,
    "joint_numerical_convergence": verify_joint_v2_gate,
    "cross_solver_spectrum_validation": verify_cross_solver_v2_gate,
}


def verify_gate_payload(
    gate: str,
    payload: dict[str, Any],
    pool: dict[str, Any],
) -> tuple[bool, str | None]:
    verifier = GATE_PAYLOAD_VERIFIERS.get(gate)
    if verifier is None:
        return False, f"no independent verifier is registered for gate {gate}"
    return verifier(payload, pool)


def verify_gate_evidence(
    policy: dict[str, Any], pool: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    stored = load_json(GATE_STATE, {}) or {}
    stored_gates = stored.get("gates", {}) if isinstance(stored, dict) else {}
    gates = {
        "pool_complete": bool(pool.get("passed")),
        "strict_pool_validation": bool(pool.get("passed")),
    }
    details: dict[str, Any] = {
        "pool_complete": {"source": "strict_pool_audit"},
        "strict_pool_validation": {"source": "strict_pool_audit"},
    }
    known = {item["gate"] for item in policy["workflow"]["actions"]}
    gate_specs = {item["gate"]: item for item in policy["workflow"]["actions"]}
    for gate in known:
        entry = stored_gates.get(gate, {})
        evidence = entry.get("evidence", []) if isinstance(entry, dict) else []
        if gate == "pool_manifest_frozen" and not evidence:
            manifest_path = STATE / "pool_manifest.json"
            if manifest_path.is_file():
                entry = {"passed": True, "checked_at": now_iso()}
                evidence = [
                    {
                        "path": str(manifest_path.relative_to(ROOT)),
                        "sha256": file_digest(manifest_path),
                    }
                ]
        valid = bool(entry.get("passed")) and bool(evidence)
        checked = []
        for item in evidence:
            path = ROOT / item.get("path", "")
            expected = str(item.get("sha256", "")).upper()
            actual = file_digest(path) if path.is_file() else None
            item_valid = bool(actual and expected and actual == expected)
            semantic_valid = True
            semantic_error = None
            if gate != "pool_manifest_frozen":
                if path.suffix.lower() != ".json":
                    semantic_valid = False
                    semantic_error = "gate evidence must be a JSON manifest"
                else:
                    try:
                        payload = load_json(path, {}) or {}
                        if not isinstance(payload, dict) or payload.get("passed") is not True:
                            semantic_valid = False
                            semantic_error = "evidence does not declare passed=true"
                        expected_version = gate_specs.get(gate, {}).get("evidence_version")
                        if expected_version and payload.get("evidence_version") != expected_version:
                            semantic_valid = False
                            semantic_error = (
                                f"evidence_version must be {expected_version}, got "
                                f"{payload.get('evidence_version')}"
                            )
                        evidence_pool_sha = payload.get("pool_sha256")
                        if not evidence_pool_sha and isinstance(payload.get("pool"), dict):
                            evidence_pool_sha = payload["pool"].get("sha256")
                        binding = gate_specs.get(gate, {}).get("binding", "pool")
                        if not evidence_pool_sha:
                            semantic_valid = False
                            semantic_error = "evidence is not bound to the audited pool SHA256"
                        elif (
                            binding == "pool"
                            and str(evidence_pool_sha).upper() != str(pool.get("sha256", "")).upper()
                        ):
                            semantic_valid = False
                            semantic_error = "evidence pool SHA256 does not match the audited pool"
                        if semantic_valid:
                            semantic_valid, semantic_error = verify_gate_payload(
                                gate, payload, pool
                            )
                    except Exception as exc:
                        semantic_valid = False
                        semantic_error = f"invalid evidence JSON: {type(exc).__name__}: {exc}"
            item_valid = item_valid and semantic_valid
            checked.append(
                {
                    "path": item.get("path"),
                    "expected_sha256": expected or None,
                    "actual_sha256": actual,
                    "semantic_passed": semantic_valid,
                    "semantic_error": semantic_error,
                    "binding": gate_specs.get(gate, {}).get("binding", "pool"),
                    "passed": item_valid,
                }
            )
            valid = valid and item_valid
        if gate == "pool_manifest_frozen" and valid:
            try:
                manifest_path = ROOT / evidence[0]["path"]
                manifest = load_json(manifest_path, {}) or {}
                valid = (
                    manifest.get("pool_sha256", "").upper() == pool.get("sha256")
                    and manifest.get("records") == pool.get("records")
                    and manifest.get("strict_validation_passed") is True
                    and manifest.get("immutable") is True
                )
            except Exception:
                valid = False
        gates[gate] = valid
        details[gate] = {
            "declared_passed": bool(entry.get("passed")) if isinstance(entry, dict) else False,
            "evidence": checked,
            "verified": valid,
            "checked_at": entry.get("checked_at") if isinstance(entry, dict) else None,
        }
    required = policy["workflow"]["required_before_training"]
    gates["training_allowed"] = all(gates.get(gate, False) for gate in required)
    details["training_allowed"] = {"required": required, "verified": gates["training_allowed"]}
    return gates, details


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def retry_or_fail(
    request: dict[str, Any], max_attempts: int, error: str, *, terminal: bool = False
) -> None:
    if not terminal and int(request["attempt"]) < max_attempts:
        request["attempt"] = int(request["attempt"]) + 1
        request["status"] = "pending"
        request["updated_at"] = now_iso()
        request["last_error"] = error
        for key in (
            "acknowledged_at",
            "claimed_at",
            "lease_expires_at",
            "finalization_grace_until",
            "worker_exit_detected_at",
            "terminal_failure",
            "failure_class",
        ):
            request.pop(key, None)
        return
    request["status"] = "failed"
    request["updated_at"] = now_iso()
    request["last_error"] = error
    request["terminal_failure"] = terminal


def build_recovery_plan(
    action: str | None, dispatch: dict[str, Any] | None, policy: dict[str, Any]
) -> dict[str, Any]:
    """Describe the next safe response without silently changing science policy."""
    if not action:
        return {
            "status": "idle",
            "automatic_retry": False,
            "recommended_strategy": "monitor",
        }

    if action == "stop_and_report":
        return {
            "status": "blocked",
            "automatic_retry": False,
            "recommended_strategy": "preserve_and_report",
            "guardrails": [
                "preserve all artifacts and checkpoints",
                "do not modify paper 1 or overwrite any pool",
                "do not start training",
            ],
        }

    dispatch = dispatch if isinstance(dispatch, dict) else {}
    status = str(dispatch.get("status", "pending"))
    request_id = dispatch.get("request_id")
    if status in {"pending", "in_progress"}:
        return {
            "status": "monitoring",
            "request_id": request_id,
            "action": action,
            "automatic_retry": False,
            "recommended_strategy": "continue_current_attempt",
        }

    if status != "failed":
        return {
            "status": "awaiting_confirmation",
            "request_id": request_id,
            "action": action,
            "automatic_retry": False,
            "recommended_strategy": "recheck_gate_evidence",
        }

    failure_class = str(dispatch.get("failure_class", "transient")).lower()
    terminal = bool(dispatch.get("terminal_failure")) or failure_class in {
        "scientific",
        "safety",
        "policy",
        "permanent",
    }
    strategies = {
        "resume_pool_generation": "validate_checkpoint_then_resume_same_command",
        "pool_validation": "revalidate_immutable_pool_and_provenance",
        "d65_colorimetry": "reproduce_colorimetry_controls_without_threshold_changes",
        "joint_numerical_convergence": "inspect_failed_geometry_then_rerun_frozen_case",
        "cross_solver_spectrum_validation": "classify_solver_disagreement_before_any_retry",
        "circular_control": "rerun_control_with_frozen_budget_and_new_output",
        "geometry_split_freeze": "audit_axis_canonicalization_and_split_manifest",
        "training_pilot": "hold_training_and_reaudit_all_required_gates",
        "closed_loop_evaluation": "recompute_frozen_evaluation_matrix",
        "paper2_result_audit": "reconcile_claims_against_immutable_evidence",
    }
    strategy = strategies.get(action, "diagnose_failure_without_protocol_changes")
    plan = {
        "status": "terminal_review" if terminal else "retries_exhausted",
        "request_id": request_id,
        "action": action,
        "failure_class": failure_class,
        "last_error": dispatch.get("last_error"),
        "automatic_retry": False,
        "recovery_owner": "independent_auditor",
        "next_action": "diagnose_repair_and_replan",
        "user_intervention_required": False,
        "recommended_strategy": strategy,
        "required_evidence": [
            "failure classification and runtime hashes",
            "reproducible repair or diagnostic result",
            "unchanged pool SHA256 and paper 1 MD5 ledger",
        ],
        "guardrails": [
            "do not change pre-registered thresholds",
            "do not overwrite or resume the old isolated pool",
            "do not modify paper 1",
            "training remains forbidden until both gate ledgers pass",
        ],
    }
    if not terminal:
        plan["retry_budget"] = int(policy["dispatch"]["max_attempts"])
    return plan


def active_executor_lease(ack: dict[str, Any], policy: dict[str, Any]) -> tuple[bool, str | None]:
    worker_pid = ack.get("worker_pid")
    if worker_pid:
        # A running ack with an explicit worker must be tied to that process.
        # Do not let an unexpired lease hide an early process exit; the next
        # supervisor pass can then perform bounded checkpoint recovery.
        if pid_alive(worker_pid):
            return True, ack.get("lease_expires_at")
        return False, None
    expires = parse_timestamp(ack.get("lease_expires_at"))
    if expires is None:
        observed = parse_timestamp(ack.get("heartbeat_at") or ack.get("observed_at"))
        if observed is not None:
            expires = observed + timedelta(
                seconds=int(policy["dispatch"].get("lease_timeout_seconds", 1800))
            )
    if expires is None:
        return False, None
    return datetime.now().astimezone() < expires, expires.isoformat(timespec="seconds")


def executor_finalization_grace(
    ack: dict[str, Any], policy: dict[str, Any], *, now: datetime | None = None
) -> tuple[bool, str | None]:
    """Allow a scheduled executor to finalize a just-finished worker."""
    if not ack.get("worker_pid") or pid_alive(ack.get("worker_pid")):
        return False, None
    checkpoint = workspace_file(ack.get("checkpoint_path"))
    if checkpoint is None or not checkpoint.is_file():
        return False, None
    current = now or datetime.now().astimezone()
    modified = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=current.tzinfo)
    grace_seconds = int(policy["dispatch"].get("finalization_grace_seconds", 1200))
    deadline = modified + timedelta(seconds=max(60, grace_seconds))
    if current >= deadline:
        return False, None
    return True, deadline.isoformat(timespec="seconds")


def select_workflow_action(policy: dict[str, Any], gates: dict[str, bool]) -> str | None:
    for item in policy["workflow"]["actions"]:
        gate = item["gate"]
        if gates.get(gate, False):
            continue
        if item.get("requires_training_allowed") and not gates.get("training_allowed"):
            return "stop_and_report"
        return item["action"]
    return None


def workspace_file(value: Any) -> Path | None:
    """Resolve a relative workspace path without allowing path escape."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        path = (ROOT / Path(value)).resolve()
        path.relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return None
    return path


def resolve_active_pool(policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a hash-backed replacement pool without mutating the frozen policy pool."""
    base = copy.deepcopy(policy["pool"])
    config = policy.get("active_pool", {})
    if not isinstance(config, dict) or config.get("enabled") is not True:
        return base, {"passed": True, "source": "policy", "active": False}
    manifest_path = workspace_file(config.get("manifest_path"))
    if manifest_path is None:
        return base, {"passed": False, "source": "manifest", "error": "active pool path escapes workspace"}
    if not manifest_path.exists():
        return base, {
            "passed": True,
            "source": "policy",
            "active": False,
            "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        }
    try:
        manifest = load_json(manifest_path, {}) or {}
        if manifest.get("schema_version") != 1 or manifest.get("active") is not True:
            raise ValueError("invalid active pool manifest schema or state")
        override = manifest.get("pool_spec")
        if not isinstance(override, dict):
            raise ValueError("active pool manifest lacks pool_spec")
        activation = manifest.get("activation_evidence", {})
        evidence_path = workspace_file(activation.get("path"))
        if evidence_path is None or not evidence_path.is_file():
            raise ValueError("activation evidence is missing")
        expected_evidence_sha = str(activation.get("sha256", "")).upper()
        if file_digest(evidence_path) != expected_evidence_sha:
            raise ValueError("activation evidence SHA256 mismatch")
        evidence = load_json(evidence_path, {}) or {}
        replacement_spec = next(
            item for item in policy["workflow"]["actions"]
            if item["gate"] == "replacement_pool_ready"
        )
        if evidence.get("schema_version") != 1 or evidence.get("passed") is not True:
            raise ValueError("activation evidence does not declare schema v1 passed=true")
        if evidence.get("evidence_version") != replacement_spec.get("evidence_version"):
            raise ValueError("activation evidence version mismatch")
        if evidence.get("pool_spec") != override:
            raise ValueError("active pool_spec differs from activation evidence")

        approved = evidence.get("approved_protocol")
        if not isinstance(approved, dict) or approved != manifest.get("approved_protocol"):
            raise ValueError("active pool approved protocol binding mismatch")
        protocol_path = workspace_file(approved.get("path"))
        if protocol_path is None or not protocol_path.is_file():
            raise ValueError("approved replacement protocol is missing")
        protocol_sha = str(approved.get("sha256", "")).upper()
        if not protocol_sha or file_digest(protocol_path) != protocol_sha:
            raise ValueError("approved replacement protocol SHA256 mismatch")
        protocol = load_json(protocol_path, {}) or {}
        if (
            protocol.get("schema_version") != 1
            or protocol.get("evidence_version") != "paper2-replacement-protocol-v1"
            or protocol.get("protocol_revision") != "v2_bound_holdout"
            or protocol.get("approved") is not True
            or protocol.get("automatic_launch_authorized") is not True
            or protocol.get("pool_spec") != override
        ):
            raise ValueError("approved replacement protocol is invalid or differs from pool_spec")
        for runtime_path, expected_sha in protocol.get("runtime_hashes", {}).items():
            path = workspace_file(runtime_path)
            if path is None or not path.is_file() or file_digest(path) != str(expected_sha).upper():
                raise ValueError(f"approved protocol runtime hash mismatch: {runtime_path}")

        source = protocol.get("source_reference_gate")
        if not isinstance(source, dict):
            raise ValueError("approved protocol lacks source reference gate")
        source_path = workspace_file(source.get("path"))
        source_sha = str(source.get("sha256", "")).upper()
        if source_path is None or not source_path.is_file() or file_digest(source_path) != source_sha:
            raise ValueError("source reference gate hash mismatch")
        source_audit = load_json(source_path, {}) or {}
        if not production_reference_audit_approved(source_audit):
            raise ValueError("source reference gate is not production-approved")
        selected = source_audit.get("approved_protocol_candidate")
        if not isinstance(selected, dict) or selected.get("passed") is not True:
            raise ValueError("source reference gate lacks the frozen candidate")
        expected_protocol = (
            int(selected["requested_nG"]),
            int(selected["Nxy"]),
            float(selected["wavelength_step_nm"]),
        )
        actual_protocol = (
            int(protocol.get("nG_requested", -1)),
            int(protocol.get("Nxy", -1)),
            float(protocol.get("wavelength_step_nm", -1.0)),
        )
        if actual_protocol != expected_protocol:
            raise ValueError("active replacement protocol differs from the v2 frozen candidate")
        registered_reference = (load_json(GATE_STATE, {}) or {}).get("gates", {}).get(
            "reference_resolution", {}
        )
        if registered_reference.get("passed") is not True or {
            "path": str(source.get("path", "")),
            "sha256": source_sha,
        } not in registered_reference.get("evidence", []):
            raise ValueError("source reference gate is not registered")

        immutable_equal = (
            "expected_records",
            "polarizations",
            "material",
            "substrate",
            "lossless",
            "range_tolerance",
            "pointwise_conservation_tolerance",
            "stored_value_tolerance",
            "quality_tolerance",
        )
        for field in immutable_equal:
            if override.get(field) != base.get(field):
                raise ValueError(f"active pool_spec changes frozen field {field}")
        if int(override.get("expected_records", 0)) <= 0:
            raise ValueError("active pool expected_records must be positive")
        expected_meta = override.get("expected_meta")
        base_meta = base.get("expected_meta", {})
        if not isinstance(expected_meta, dict):
            raise ValueError("active pool expected_meta is missing")
        for field in (
            "seed", "material", "substrate", "background", "pols", "n_samples",
            "sampler_version", "quality_rule",
        ):
            if expected_meta.get(field) != base_meta.get(field):
                raise ValueError(f"active pool expected_meta changes frozen field {field}")
        required = set(base.get("required_record_fields", []))
        if not required.issubset(set(override.get("required_record_fields", []))):
            raise ValueError("active pool removes required record fields")

        spec = copy.deepcopy(override)
        pool_path = workspace_file(spec.get("path"))
        if pool_path is None or not pool_path.is_file():
            raise ValueError("activated pool is missing or outside workspace")
        replacement_root = (ROOT / "data" / "replacement").resolve()
        try:
            pool_path.relative_to(replacement_root)
        except ValueError as exc:
            raise ValueError("active pool must be a new file below data/replacement") from exc
        actual_pool_sha = file_digest(pool_path)
        expected_pool_sha = str(manifest.get("pool_sha256", "")).upper()
        if actual_pool_sha != expected_pool_sha:
            raise ValueError("activated pool SHA256 mismatch")
        if str(evidence.get("pool_sha256", "")).upper() != actual_pool_sha:
            raise ValueError("activation evidence is not bound to the replacement pool")
        activation_id = hashlib.sha256(
            f"{protocol_sha}|{actual_pool_sha}".encode("ascii")
        ).hexdigest()[:24]
        if evidence.get("activation_id") != activation_id or manifest.get("activation_id") != activation_id:
            raise ValueError("active pool activation_id is not hash-bound")
        previous = manifest.get("previous_pool", {})
        base_path = workspace_file(base.get("path"))
        if base_path is None or not base_path.is_file():
            raise ValueError("policy source pool is unavailable")
        if str(previous.get("path", "")).replace("\\", "/") != str(base["path"]).replace("\\", "/"):
            raise ValueError("active pool previous path does not match policy pool")
        if str(previous.get("sha256", "")).upper() != file_digest(base_path):
            raise ValueError("active pool previous SHA256 does not match policy pool")
        if str(previous.get("md5", "")).upper() != file_digest(base_path, "md5"):
            raise ValueError("active pool previous MD5 does not match policy pool")
        pool_manifest = manifest.get("pool_manifest")
        if not isinstance(pool_manifest, dict):
            raise ValueError("active pool lacks strict pool manifest binding")
        pool_manifest_path = workspace_file(pool_manifest.get("path"))
        pool_manifest_sha = str(pool_manifest.get("sha256", "")).upper()
        if (
            pool_manifest_path is None
            or not pool_manifest_path.is_file()
            or file_digest(pool_manifest_path) != pool_manifest_sha
        ):
            raise ValueError("strict replacement pool manifest hash mismatch")
        strict_manifest = load_json(pool_manifest_path, {}) or {}
        if (
            strict_manifest.get("strict_validation_passed") is not True
            or strict_manifest.get("immutable") is not True
            or str(strict_manifest.get("pool_sha256", "")).upper() != actual_pool_sha
            or int(strict_manifest.get("records", -1)) != int(spec["expected_records"])
            or strict_manifest.get("approved_protocol") != approved
            or strict_manifest.get("activation_id") != activation_id
            or str(strict_manifest.get("pool_spec_sha256", "")).upper()
            != json_payload_digest(spec)
        ):
            raise ValueError("strict replacement pool manifest content mismatch")
        return spec, {
            "passed": True,
            "source": "manifest",
            "active": True,
            "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": file_digest(manifest_path),
            "pool_path": str(pool_path.relative_to(ROOT)).replace("\\", "/"),
            "pool_sha256": actual_pool_sha,
            "previous_pool": previous,
            "activation_evidence": activation,
        }
    except Exception as exc:
        return base, {
            "passed": False,
            "source": "manifest",
            "active": False,
            "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "error": f"{type(exc).__name__}: {exc}",
        }


def validate_completed_ack(
    ack: dict[str, Any],
    pool_sha256: str | None,
    policy: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Require a durable, hash-backed completion handoff before advancing."""
    checks = ack.get("checks")
    if not isinstance(checks, dict):
        return False, "completed ack missing checks object"
    declared_pool = str(checks.get("pool_sha256", "")).upper()
    if not declared_pool or declared_pool != str(pool_sha256 or "").upper():
        return False, "completed ack pool SHA256 does not match audited pool"

    outputs = ack.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return False, "completed ack outputs must be a non-empty object list"
    for item in outputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            return False, "completed ack outputs must contain path/material objects"
        if not isinstance(item.get("material"), str) or not item["material"].strip():
            return False, "completed ack output material is missing"
        path = workspace_file(item["path"])
        if path is None or not path.is_file():
            return False, f"completed ack output is missing: {item.get('path')}"
        expected_sha = str(item.get("sha256", "")).upper()
        if (
            len(expected_sha) != 64
            or any(char not in "0123456789ABCDEF" for char in expected_sha)
            or file_digest(path) != expected_sha
        ):
            return False, f"completed ack output SHA256 mismatch: {item.get('path')}"
        if policy is not None:
            protected = {
                str(asset.get("path", "")).replace("\\", "/").casefold()
                for asset in (
                    *policy.get("protected_files", []),
                    *policy.get("immutable_assets", []),
                )
            }
            current_pool = str(policy.get("pool", {}).get("path", "")).replace(
                "\\", "/"
            ).casefold()
            if current_pool:
                protected.add(current_pool)
            relative = str(path.relative_to(ROOT)).replace("\\", "/").casefold()
            if relative in protected:
                return False, f"completed ack output targets immutable asset: {item.get('path')}"

    paper_hashes = ack.get("paper_hashes")
    if not isinstance(paper_hashes, list) or not paper_hashes:
        return False, "completed ack paper_hashes must be a non-empty object list"
    seen_paper_paths = set()
    for item in paper_hashes:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            return False, "completed ack paper_hashes must contain path/md5 objects"
        expected = str(item.get("md5", "")).upper()
        if len(expected) != 32 or any(char not in "0123456789ABCDEF" for char in expected):
            return False, f"completed ack has invalid MD5: {item.get('path')}"
        path = workspace_file(item["path"])
        if path is None or not path.is_file():
            return False, f"completed ack paper hash file is missing: {item.get('path')}"
        actual = file_digest(path, "md5")
        if actual != expected:
            return False, f"completed ack paper hash mismatch: {item.get('path')}"
        seen_paper_paths.add(str(path.relative_to(ROOT)).replace("\\", "/").casefold())
    if policy is not None and policy.get("protected_files"):
        required_paper_paths = {
            str(item.get("path", "")).replace("\\", "/").casefold()
            for item in policy["protected_files"]
        }
        if seen_paper_paths != required_paper_paths:
            return False, "completed ack paper_hashes must cover exactly the protected paper files"
    return True, None


def validate_failed_ack(
    ack: dict[str, Any], pool_sha256: str | None
) -> tuple[bool, str | None]:
    checks = ack.get("checks")
    if not isinstance(checks, dict) or str(checks.get("pool_sha256", "")).upper() != str(
        pool_sha256 or ""
    ).upper():
        return False, "failed ack pool SHA256 does not match audited pool"
    evidence = ack.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return False, "failed ack requires hash-backed evidence"
    for item in evidence:
        valid, error = verify_file_binding(item, "failed ack evidence")
        if not valid:
            return False, error
    return True, None


def build_instruction(action: str, policy: dict[str, Any]) -> str:
    protected = ", ".join(item["path"] for item in policy["protected_files"])
    immutable = ", ".join(item["path"] for item in policy.get("immutable_assets", []))
    global_guard = (
        " Never modify, overwrite, resume, or emit outputs to any protected paper file, immutable legacy "
        f"asset, old pool, or paper 1 script. Protected papers: {protected}. Immutable assets: {immutable}."
    )
    if action == "resume_pool_generation":
        return (
            f"Resume exactly: {policy['pool']['resume_command']}. Do not create a new pool or start training. "
            "Write an atomic executor ack. On completed, use outputs=[{path,material}], "
            "paper_hashes=[{path,md5}], and checks.pool_sha256 matching the audited pool."
            + global_guard
        )
    if action == "pool_validation":
        return (
            "Re-run strict validation, create an immutable manifest with pool hash/provenance, and atomically "
            "acknowledge this request. Do not edit the pool or start training. D65 colorimetry, joint numerical "
            "convergence, cross-solver spectra, circular control, and geometry split remain mandatory. "
            f"Protected files: {protected}. On completed, use outputs=[{{path,material}}], "
            "paper_hashes=[{path,md5}], and checks.pool_sha256 matching the audited pool."
            + global_guard
        )
    if action == "stop_and_report":
        return "Stop all downstream work and report the first failing check."
    action_spec = next(
        (item for item in policy["workflow"]["actions"] if item["action"] == action),
        {},
    )
    gate = action_spec.get("gate")
    action_instructions = {
        "d65_colorimetry": (
            "Implement a versioned paper 2 D65 SPD colorimetry path without changing legacy paper 1 results. "
            "Lab must be computed directly from unclipped XYZ; sRGB is display-only. Verify a perfect reflector "
            "maps to neutral D65 white and archive tests plus derived-label provenance."
        ),
        "joint_numerical_convergence": (
            f"Run {action_spec.get('runner', 'the configured joint-convergence runner')}. "
            "Require the activated replacement pool, its exact SHA256, the auditor-approved production protocol, "
            "and the frozen 32-geometry raw reference. Recompute D65 labels from raw R/T, handle canonical-axis p/s "
            "mapping, and reject all historical v1/v1.1 evidence."
        ),
        "reference_resolution": (
            f"Run {action_spec.get('runner', 'the configured reference audit')}. "
            "Use the initial eight cases only to freeze one candidate. Register this gate only after the pre-frozen "
            "24-case confirmation passes against the nG450/Nxy768/0.5nm final reference, with exactly 240 tasks "
            "or 288 when the frozen candidate is outside the five-protocol reference matrix. "
            "Never mark the historical joint gate passed or reuse coarse labels for a fine-grid claim."
        ),
        "replacement_pool_generation": (
            f"Run {action_spec.get('runner', 'the configured replacement-pool runner')}. "
            "Use only the hash-bound protocol approved by the independent 32-case audit. Generate a canonical-axis, "
            "dual-polarization pool below data/replacement with SQLite WAL resume and correct D65 labels. Never use "
            "the historical generator, overwrite the nG131 pool, or activate the result from the executor task."
        ),
        "cross_solver_spectrum_validation": (
            f"Run {action_spec.get('runner', 'the configured cross-solver runner')}. "
            "Bind the matched third-party comparison to the active pool and approved nG/Nxy protocol. Use 12 frozen "
            "sharpness-stratified geometries, both polarizations, analytic and symmetry controls, plus independent "
            "order-axis, grid-axis, and corner stress configurations on four cases. Reject old-pool or v1 evidence."
        ),
        "circular_control": (
            "Generate or validate a corrected-solver, air-background circular TiO2 control under a matched, frozen "
            "budget. Never overwrite the elliptical pool or any paper 1 pool."
        ),
        "geometry_split_freeze": (
            "Canonicalize long/short axes with the required p/s swap, assign stable geometry_id values, and freeze "
            "geometry-level train/validation/test splits. Do not use label-preserving geometry jitter."
        ),
        "training_pilot": (
            "Run only a bounded dual-spectrum pilot after independently confirming training_allowed=true. Keep p/s "
            "paired by geometry and archive configs, seeds, checkpoints, and holdout metrics."
        ),
        "closed_loop_evaluation": (
            "Run the frozen circular/elliptical, naive/top-K/random-K and joint-polarization evaluation matrix. Do not "
            "change thresholds after observing results."
        ),
        "paper2_result_audit": (
            "Independently audit all paper 2 claims against immutable artifacts and report null or narrowed results "
            "when pre-registered claims fail. Do not modify paper 1."
        ),
    }
    instruction = action_instructions.get(action, f"Execute the bounded pipeline action: {action}.")
    if action_spec.get("diagnostic_only") is True:
        return (
            instruction
            + " Preserve a versioned result and checkpoint, but do not register the gate or activate the new pool. "
            "After successful computation, write executor_ack status=failed with failure_class=scientific and point "
            "to the result so the independent auditor can validate and atomically activate it."
            + global_guard
        )
    auditor_instruction = ""
    if action_spec.get("auditor"):
        auditor_instruction = (
            f" After the worker finishes, run {action_spec['auditor']} as a separate independent reproduction. "
            "Register only the auditor JSON; never register the worker evidence directly."
        )
    return (
        instruction
        + auditor_instruction
        + f" On success, write a versioned evidence artifact, register gate {gate} in .state/gate_state.json "
        "with its SHA256, then atomically write executor_ack.json with the matching request_id and attempt. "
        "The gate evidence manifest must be JSON, declare passed=true, and bind to the requested pool SHA256. "
        "Before long-running work, write a running ack with a renewable lease and optional worker_pid. "
        "On completed, use outputs=[{path,material}], paper_hashes=[{path,md5}], and "
        "checks.pool_sha256 matching the audited pool; the supervisor recomputes every file hash. "
        "On scientific failure, do not mark the gate passed; write a failed ack with evidence."
        + global_guard
    )


def make_dispatch_id(
    stage: str,
    action: str,
    artifact_sha256: str | None,
    strategy_revision: int = 0,
    strategy_decision: str | None = None,
    strategy_based_on: str | None = None,
) -> str:
    suffix = ""
    if strategy_revision:
        suffix = (
            f"|strategy:{strategy_revision}|decision:{strategy_decision or 'none'}"
            f"|based_on:{strategy_based_on or 'none'}"
        )
    raw = f"{stage}|{action}|{artifact_sha256 or 'none'}{suffix}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def archive_dispatch(request: dict[str, Any], ack: dict[str, Any], next_action: str) -> Path | None:
    """Write one durable terminal handoff snapshot before replacing a request."""
    request_id = str(request.get("request_id", ""))
    if not request_id or any(char not in "0123456789abcdefABCDEF-" for char in request_id):
        return None
    attempt = int(request.get("attempt", 0))
    history = STATE / "dispatch_history"
    path = history / f"{request_id}-attempt{attempt}.json"
    matching_ack = (
        ack
        if ack.get("request_id") == request_id
        and int(ack.get("attempt", 0)) == attempt
        else None
    )
    payload = {
        "schema_version": 1,
        "request": request,
        "final_ack": matching_ack,
        "next_action": next_action,
    }
    if path.exists():
        existing = load_json(path, {}) or {}
        if existing != payload:
            raise ValueError(f"dispatch history collision: {path}")
    else:
        atomic_json(path, payload)
    return path


def strategy_override(
    action: str, policy: dict[str, Any], existing: dict[str, Any]
) -> dict[str, Any] | None:
    strategy = policy.get("strategy_override")
    if not isinstance(strategy, dict) or strategy.get("enabled") is not True:
        return None
    decision = strategy.get("decision")
    if strategy.get("action") != action or decision not in {
        "retry_same_gate", "transition_after_failure"
    }:
        return None
    try:
        revision = int(strategy.get("revision", 0))
    except (TypeError, ValueError):
        return None
    if revision <= 0:
        return None
    same_strategy_request = (
        existing.get("status") in {"pending", "in_progress", "acknowledged"}
        and
        existing.get("action") == action
        and int(existing.get("strategy_revision", 0)) == revision
        and existing.get("strategy_based_on") == strategy.get("based_on_request_id")
    )
    if not same_strategy_request:
        if (
            existing.get("status") != "failed"
            or existing.get("request_id") != strategy.get("based_on_request_id")
        ):
            return None
        attempts_exhausted = int(existing.get("attempt", 0)) >= int(
            existing.get("max_attempts", policy["dispatch"]["max_attempts"])
        )
        if not existing.get("terminal_failure") and not attempts_exhausted:
            return None
        previous_revision = int(existing.get("strategy_revision", 0))
        if revision <= previous_revision:
            return None
        previous_action = existing.get("action")
        if decision == "retry_same_gate" and action != previous_action:
            return None
        if decision == "transition_after_failure" and (
            action == previous_action or strategy.get("from_action") != previous_action
        ):
            return None
    evidence = strategy.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return None
    for item in evidence:
        if not isinstance(item, dict):
            return None
        path = workspace_file(item.get("path"))
        expected = str(item.get("sha256", "")).upper()
        if path is None or not path.is_file() or not expected or file_digest(path) != expected:
            return None
    instruction_append = strategy.get("instruction_append", "")
    if not isinstance(instruction_append, str) or not instruction_append.strip() or len(instruction_append) > 4000:
        return None
    return strategy


def update_dispatch(action: str, policy: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    existing = load_json(DISPATCH_REQUEST, {}) or {}
    ack = load_json(EXECUTOR_ACK, {}) or {}
    replacement_gate_activated = bool(
        existing.get("status") == "failed"
        and existing.get("action") == "replacement_pool_generation"
        and existing.get("terminal_failure") is True
        and str(existing.get("failure_class", "")).lower() == "scientific"
        and audit.get("training_gates", {}).get("replacement_pool_ready") is True
    )
    ack_is_terminal_for_existing = (
        ack.get("request_id") == existing.get("request_id")
        and int(ack.get("attempt", 0)) == int(existing.get("attempt", 0))
        and ack.get("status") in {"completed", "succeeded", "failed"}
    )
    live_request_immutable = bool(
        existing.get("status") in {"pending", "in_progress"}
        and existing.get("action")
        and not ack_is_terminal_for_existing
    )
    terminal_ack_transition = bool(
        existing.get("status") in {"pending", "in_progress"}
        and existing.get("action")
        and ack_is_terminal_for_existing
    )
    preserve_existing = live_request_immutable or terminal_ack_transition
    waiting_for_replan = False
    if preserve_existing:
        # A durable non-terminal request is immutable until it is acknowledged
        # or fails. Workflow, policy, and active-pool revisions apply only to
        # the next request and must never orphan a live worker/ack pair.
        action = str(existing["action"])
        request = existing
        request_id = str(existing["request_id"])
        pool_sha = existing.get("payload", {}).get("pool_sha256") or audit.get("pool", {}).get("sha256")
        max_attempts = int(existing.get("max_attempts", policy["dispatch"]["max_attempts"]))
    else:
        pool_sha = audit.get("pool", {}).get("sha256")
        strategy_action = action
        configured_strategy = policy.get("strategy_override", {})
        if existing.get("status") == "failed" and isinstance(configured_strategy, dict):
            strategy_action = str(configured_strategy.get("action") or action)
        strategy = strategy_override(strategy_action, policy, existing)
        if strategy is not None:
            action = strategy_action
        if (
            existing.get("status") == "failed"
            and strategy is None
            and not replacement_gate_activated
        ):
            # Scientific, safety, policy, permanent, and exhausted transient
            # failures remain terminal until an evidence-backed strategy names
            # the exact failed request. Never advance to another gate merely
            # because the workflow selector now points at it.
            waiting_for_replan = True
            action = str(existing.get("action") or action)
            request = existing
            request_id = str(existing["request_id"])
            pool_sha = existing.get("payload", {}).get("pool_sha256") or pool_sha
            max_attempts = int(existing.get("max_attempts", policy["dispatch"]["max_attempts"]))
        else:
            strategy_revision = int(strategy.get("revision", 0)) if strategy else 0
            request_id = make_dispatch_id(
                "paper2_pipeline",
                action,
                pool_sha,
                strategy_revision,
                strategy.get("decision") if strategy else None,
                strategy.get("based_on_request_id") if strategy else None,
            )
            max_attempts = int(policy["dispatch"]["max_attempts"])
    timeout_seconds = int(
        policy["dispatch"].get(
            "pickup_timeout_seconds", policy["dispatch"].get("ack_timeout_seconds", 1800)
        )
    )

    if preserve_existing or waiting_for_replan:
        pass
    elif existing.get("request_id") == request_id:
        request = existing
        request["protocol_version"] = 2
        instruction = build_instruction(action, policy)
        if strategy:
            instruction += " Strategy amendment: " + strategy["instruction_append"].strip()
            request["strategy_revision"] = strategy_revision
            request["strategy_based_on"] = strategy["based_on_request_id"]
            request["strategy_evidence"] = strategy["evidence"]
        request["instruction"] = instruction
    else:
        if existing.get("request_id") and (
            ack_is_terminal_for_existing
            or existing.get("status") in {"acknowledged", "failed"}
        ):
            archive_dispatch(existing, ack, action)
        timestamp = now_iso()
        instruction = build_instruction(action, policy)
        if strategy:
            instruction += " Strategy amendment: " + strategy["instruction_append"].strip()
        request = {
            "schema_version": 1,
            "protocol_version": 2,
            "request_id": request_id,
            "target_thread_id": policy["executor_thread_id"],
            "stage": "paper2_pipeline",
            "action": action,
            "status": "pending",
            "attempt": 1,
            "max_attempts": max_attempts,
            "created_at": timestamp,
            "updated_at": timestamp,
            "ack_required": True,
            "payload": {
                "pool": policy["pool"]["path"],
                "pool_sha256": pool_sha,
                "audit_result": str(AUDIT_RESULT.relative_to(ROOT)),
                "next_plan": str(NEXT_PLAN.relative_to(ROOT)),
            },
            "instruction": instruction,
        }
        if strategy:
            request["strategy_revision"] = strategy_revision
            request["strategy_based_on"] = strategy["based_on_request_id"]
            request["strategy_evidence"] = strategy["evidence"]

    same_request_attempt = (
        ack.get("request_id") == request_id
        and int(ack.get("attempt", 0)) == int(request["attempt"])
    )
    ack_thread = ack.get("thread_id") or ack.get("target_thread_id")
    identity_mismatch = same_request_attempt and ack_thread != policy["executor_thread_id"]
    matching_ack = same_request_attempt and not identity_mismatch
    if same_request_attempt and identity_mismatch:
        request["failure_class"] = "policy"
        retry_or_fail(
            request,
            max_attempts,
            "executor thread identity mismatch",
            terminal=True,
        )
    if matching_ack and request.get("status") not in {"failed", "acknowledged"}:
        ack_status = ack.get("status")
        if ack_status in {"accepted", "claimed", "running", "in_progress"}:
            lease_active, lease_expires_at = active_executor_lease(ack, policy)
            if lease_active:
                request["status"] = "in_progress"
                request["claimed_at"] = ack.get("observed_at", request.get("claimed_at", now_iso()))
                request["lease_expires_at"] = lease_expires_at
            else:
                grace_active, grace_until = executor_finalization_grace(ack, policy)
                if grace_active:
                    request["status"] = "in_progress"
                    request["worker_exit_detected_at"] = request.get(
                        "worker_exit_detected_at", now_iso()
                    )
                    request["finalization_grace_until"] = grace_until
                else:
                    retry_or_fail(request, max_attempts, "executor lease expired")
        elif ack_status in {"completed", "succeeded"}:
            if action == "stop_and_report":
                request["status"] = "acknowledged"
                request["acknowledged_at"] = ack.get("observed_at", now_iso())
            else:
                valid_ack, ack_error = validate_completed_ack(ack, pool_sha, policy)
                if valid_ack:
                    request["status"] = "acknowledged"
                    request["acknowledged_at"] = ack.get("observed_at", now_iso())
                else:
                    retry_or_fail(
                        request,
                        max_attempts,
                        ack_error or "invalid completed ack",
                    )
        elif ack_status == "failed":
            failure_class = str(ack.get("failure_class", "transient")).lower()
            valid_failure, failure_error = validate_failed_ack(ack, pool_sha)
            if not valid_failure:
                failure_class = "policy"
            terminal = failure_class in {"scientific", "safety", "policy", "permanent"}
            request["failure_class"] = failure_class
            retry_or_fail(
                request,
                max_attempts,
                failure_error or ack.get("error", "executor reported failure"),
                terminal=terminal,
            )
    if request.get("status") == "acknowledged" and action not in {"resume_pool_generation", "stop_and_report"}:
        action_spec = next(
            (
                item
                for item in policy["workflow"]["actions"]
                if item.get("action") == action
            ),
            {},
        )
        action_gate = action_spec.get("gate")
        gate_verified = bool(action_gate and audit.get("training_gates", {}).get(action_gate))
        if not gate_verified:
            retry_or_fail(request, max_attempts, "acknowledged without verified gate evidence")
    elif request.get("status") in {"pending", "in_progress"} and not matching_ack:
        updated = parse_timestamp(request.get("updated_at")) or datetime.now().astimezone()
        age = (datetime.now().astimezone() - updated).total_seconds()
        if age >= timeout_seconds:
            retry_or_fail(request, max_attempts, "executor pickup timeout")

    atomic_json(DISPATCH_REQUEST, request)
    atomic_json(
        LEGACY_INBOX,
        {
            "from": "pipeline_supervisor",
            "request_id": request["request_id"],
            "attempt": request["attempt"],
            "created_at": request.get("updated_at") or now_iso(),
            "priority": "urgent" if action == "stop_and_report" else "normal",
            "action": action,
            "instruction": request["instruction"],
            "audit_result": str(AUDIT_RESULT.relative_to(ROOT)),
            "ack_required": True,
        },
    )
    return request


def evaluate_once(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or load_policy()
    active_pool_spec, active_pool = resolve_active_pool(policy)
    runtime_policy = copy.deepcopy(policy)
    runtime_policy["pool"] = active_pool_spec
    if active_pool.get("active") is True:
        previous = active_pool.get("previous_pool", {})
        runtime_policy.setdefault("immutable_assets", []).append(
            {"path": previous["path"], "md5": previous["md5"]}
        )
    status = load_json(STATUS, {}) or {}
    watchdog = load_json(STATE / "paper2_watchdog_status.json", {}) or {}
    producer_status = status.get("status", "missing")
    alive = pid_alive(status.get("pid") or status.get("python_pid"))
    pool = audit_pool(ROOT / runtime_policy["pool"]["path"], runtime_policy["pool"])
    protected = audit_protected_files(runtime_policy)
    protected_passed = all(item["passed"] for item in protected)
    integrity = verify_policy_integrity(policy)
    errors: list[dict[str, Any]] = []
    action = None
    effective_status = producer_status
    reconciled = False

    if not integrity.get("passed", False):
        effective_status = "blocked"
        errors.append({"code": "POLICY_INTEGRITY_MISMATCH", "detail": integrity})
        action = "stop_and_report"
    if not active_pool.get("passed", False):
        effective_status = "blocked"
        errors.append({"code": "ACTIVE_POOL_MANIFEST_INVALID", "detail": active_pool})
        action = "stop_and_report"

    if action is not None:
        pass
    elif producer_status == "failed":
        effective_status = "failed"
        errors.append({"code": "PRODUCER_REPORTED_FAILED"})
        action = "stop_and_report"
    elif pool["passed"] and protected_passed and not alive:
        effective_status = "completed"
        reconciled = producer_status != "completed"
    elif alive:
        effective_status = "running"
    elif pool["records"] < pool["expected_records"] and pool["healthy_checkpoint"]:
        effective_status = "interrupted"
        if recovery_attempt(status) < int(policy["dispatch"]["max_attempts"]):
            action = "resume_pool_generation"
        else:
            errors.append({"code": "RECOVERY_ATTEMPTS_EXHAUSTED"})
            action = "stop_and_report"
    else:
        effective_status = "blocked"
        errors.extend(pool["errors"][:20])
        if not protected_passed:
            errors.append({"code": "PROTECTED_FILE_HASH_MISMATCH"})
        action = "stop_and_report"

    stage_passed = effective_status == "completed" and pool["passed"] and protected_passed
    training_gates, gate_evidence = verify_gate_evidence(runtime_policy, pool)
    if stage_passed and action is None:
        action = select_workflow_action(runtime_policy, training_gates)
    audit = {
        "schema_version": 2,
        "stage": "pool_generation",
        "producer_status": producer_status,
        "effective_status": effective_status,
        "producer_pid_alive": alive,
        "status_reconciled": reconciled,
        "passed": stage_passed,
        "protected_files": protected,
        "policy_integrity": integrity,
        "active_pool": active_pool,
        "pool": pool,
        "training_gates": training_gates,
        "gate_evidence": gate_evidence,
        "watchdog": watchdog if isinstance(watchdog, dict) else {},
        "errors": errors,
        "generated_at": now_iso(),
    }
    atomic_json(AUDIT_RESULT, audit)

    integrity_blocked = not integrity.get("passed", False)
    if integrity_blocked:
        # Keep the last durable request intact while policy and lock are being
        # atomically revised. Replacing it with stop_and_report would orphan a
        # live worker whose acknowledgement is bound to the existing request.
        dispatch = load_json(DISPATCH_REQUEST, None)
    else:
        dispatch = update_dispatch(action, runtime_policy, audit) if action else None
    if dispatch and dispatch.get("status") == "failed":
        action = "stop_and_report"
        stage_passed = False
    recovery_plan = build_recovery_plan(
        dispatch.get("action") if isinstance(dispatch, dict) else action,
        dispatch,
        runtime_policy,
    )
    if isinstance(dispatch, dict) and dispatch.get("status") in {"pending", "in_progress", "failed"}:
        active_stage = dispatch.get("action")
    elif effective_status == "completed":
        active_stage = "pool_generation_complete"
    else:
        active_stage = "pool_generation"

    dispatch_status = dispatch.get("status") if isinstance(dispatch, dict) else None
    pipeline_action = action
    if action == "stop_and_report":
        pipeline_status = "blocked"
        pipeline_action = "stop_and_report"
    elif dispatch_status in {"pending", "in_progress"}:
        pipeline_status = "running" if dispatch_status == "in_progress" else "pending"
        pipeline_action = dispatch.get("action") or action
    elif dispatch_status == "failed":
        pipeline_status = "blocked"
        pipeline_action = "stop_and_report"
    else:
        pipeline_status = effective_status
    pipeline_complete = pipeline_status == "completed" and pipeline_action is None

    audit["active_stage"] = active_stage
    audit["dispatch"] = dispatch
    audit["recovery_plan"] = recovery_plan
    audit["pipeline_status"] = pipeline_status
    audit["pipeline_complete"] = pipeline_complete
    atomic_json(AUDIT_RESULT, audit)
    next_plan = {
        "schema_version": 2,
        "audit_passed": stage_passed,
        "pipeline_status": pipeline_status,
        "pipeline_complete": pipeline_complete,
        "recommended_next": pipeline_action or "monitor_existing_pool",
        "dispatch_request": str(DISPATCH_REQUEST.relative_to(ROOT)) if dispatch else None,
        "dispatch_status": dispatch_status,
        "training_allowed": training_gates["training_allowed"],
        "scientific_blockers": [key for key, value in training_gates.items() if not value and key != "training_allowed"],
        "recovery": recovery_plan,
        "generated_at": now_iso(),
    }
    atomic_json(NEXT_PLAN, next_plan)
    controller = {
        "schema_version": 1,
        "project": runtime_policy["project"],
        "controller_status": pipeline_status,
        "producer_status": producer_status,
        "effective_status": effective_status,
        "pipeline_status": pipeline_status,
        "pipeline_complete": pipeline_complete,
        "status_reconciled": reconciled,
        "current_stage": active_stage,
        "active_stage": active_stage,
        "next_action": pipeline_action,
        "dispatch": dispatch,
        "recovery_plan": recovery_plan,
        "training_allowed": training_gates["training_allowed"],
        "watchdog": watchdog if isinstance(watchdog, dict) else {},
        "updated_at": now_iso(),
    }
    atomic_json(CONTROLLER_STATE, controller)
    return controller


def watch(interval: int) -> None:
    lock_handle = acquire_single_instance_lock(SUPERVISOR_LOCK)
    if lock_handle is None:
        print(
            json.dumps(
                {
                    "controller_status": "already_running",
                    "lock": str(SUPERVISOR_LOCK),
                    "updated_at": now_iso(),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        return
    last_fingerprint = None
    try:
        while True:
            try:
                policy = load_policy()
                paths = [STATUS, POLICY, EXECUTOR_ACK, GATE_STATE, ROOT / policy["pool"]["path"]]
                parts = []
                for path in paths:
                    stat = path.stat() if path.exists() else None
                    parts.append((str(path), stat.st_size if stat else None, stat.st_mtime_ns if stat else None))
                status = load_json(STATUS, {}) or {}
                parts.append(("pid_alive", pid_alive(status.get("pid") or status.get("python_pid"))))
                parts.append(("minute", int(time.time() // 60)))
                fingerprint = repr(parts)
                if fingerprint != last_fingerprint:
                    controller = evaluate_once(policy)
                    transition = run_auto_transition(controller)
                    payload = {"controller": controller, "auto_transition": transition}
                    print(json.dumps(payload, ensure_ascii=True), flush=True)
                    last_fingerprint = None if (
                        isinstance(transition, dict)
                        and transition.get("status") == "advanced"
                    ) else fingerprint
            except Exception as exc:
                failure = {
                    "schema_version": 1,
                    "controller_status": "blocked",
                    "next_action": "stop_and_report",
                    "training_allowed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": now_iso(),
                }
                atomic_json(CONTROLLER_STATE, failure)
                print(json.dumps(failure, ensure_ascii=True), flush=True)
            time.sleep(max(5, interval))
    finally:
        lock_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    if args.watch:
        watch(args.interval)
    else:
        print(json.dumps(evaluate_once(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
