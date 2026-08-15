#!/usr/bin/env python3
"""Generate a hash-bound, versioned paper 2 replacement pool.

The scientific configuration comes only from an auditor-approved protocol.
Results are committed one task at a time to SQLite WAL storage; the final
pickle is published atomically only after every task and the strict pool audit
pass.  Retrying never changes nG, Nxy, wavelength grid, or material settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import sys
import time
from contextlib import AbstractContextManager
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper2_colorimetry_fine as colorimetry  # noqa: E402
import pipeline_supervisor as supervisor  # noqa: E402
from rcwa_batch import generate_params_elliptical, rcwa_spectrum  # noqa: E402
from scripts.run_joint_convergence import retained_order  # noqa: E402


PROTOCOL_VERSION = "paper2-replacement-protocol-v1"
EVIDENCE_VERSION = "paper2-replacement-pool-v1"
RUNNER_VERSION = "paper2-replacement-runner-v1"
ALLOWED_OUTPUT_DIRECTORY = (ROOT / "data" / "replacement").resolve()
BASE_REQUIRED_FIELDS = {
    "L", "W", "H", "P", "r", "pol", "material", "substrate",
    "nG_actual", "retry_nG", "isolated", "wl_nm", "R", "T",
    "R_plus_T_mean", "quality_pass", "success",
}
REPLACEMENT_REQUIRED_FIELDS = {
    "geometry_id", "background", "nG_requested", "nG_retained", "Nxy",
    "wavelength_step_nm", "pointwise_conservation_error_max", "xyz", "lab",
    "srgb_display", "label_provenance_version", "attempts",
}


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def canonical_workspace_path(raw: str, *, require_replacement_dir: bool = False) -> Path:
    candidate = (ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside the workspace: {raw}") from exc
    if require_replacement_dir:
        try:
            candidate.relative_to(ALLOWED_OUTPUT_DIRECTORY)
        except ValueError as exc:
            raise ValueError("replacement output must be below data/replacement") from exc
        if candidate.suffix.lower() != ".pkl":
            raise ValueError("replacement output must use a .pkl filename")
    return candidate


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            transient = (
                isinstance(exc, PermissionError)
                or getattr(exc, "winerror", None) in {5, 32}
                or exc.errno in {13, 16}
            )
            if not transient or attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock(AbstractContextManager["RunLock"]):
    """Exclusive process lock with safe stale-owner recovery."""

    def __init__(self, path: Path, protocol_sha256: str):
        self.path = path
        self.protocol_sha256 = protocol_sha256
        self.token = hashlib.sha256(
            f"{os.getpid()}|{time.time_ns()}|{protocol_sha256}".encode("utf-8")
        ).hexdigest()

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(3):
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                try:
                    owner = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    owner = {}
                if pid_alive(int(owner.get("pid", -1))):
                    raise RuntimeError(
                        f"replacement pool runner already active as PID {owner.get('pid')}"
                    )
                self.path.unlink(missing_ok=True)
                continue
            payload = {
                "pid": os.getpid(),
                "token": self.token,
                "protocol_sha256": self.protocol_sha256,
                "started_at": supervisor.now_iso(),
            }
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            return self
        raise RuntimeError("could not recover stale replacement pool lock")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            owner = {}
        if owner.get("token") == self.token:
            self.path.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def protected_paths(policy: dict[str, Any]) -> set[Path]:
    items = (*policy.get("protected_files", []), *policy.get("immutable_assets", []))
    return {canonical_workspace_path(str(item["path"])) for item in items}


def source_gate_matches(protocol: dict[str, Any]) -> None:
    source = protocol.get("source_reference_gate", {})
    source_path = canonical_workspace_path(str(source.get("path", "")))
    if not source_path.is_file() or file_digest(source_path) != str(source.get("sha256", "")).upper():
        raise ValueError("approved protocol reference evidence hash mismatch")
    source_evidence = load_json(source_path)
    if not supervisor.production_reference_audit_approved(source_evidence):
        raise ValueError("approved protocol reference evidence is not a v2-bound pass")
    selected = source_evidence.get("approved_protocol_candidate")
    if not isinstance(selected, dict) or selected.get("passed") is not True:
        raise ValueError("approved reference evidence lacks the frozen candidate")
    expected = (
        int(selected["requested_nG"]),
        int(selected["Nxy"]),
        float(selected["wavelength_step_nm"]),
    )
    actual = (
        int(protocol.get("nG_requested", -1)),
        int(protocol.get("Nxy", -1)),
        float(protocol.get("wavelength_step_nm", -1.0)),
    )
    if actual != expected:
        raise ValueError("replacement protocol differs from the v2 frozen candidate")
    gate_state = supervisor.load_json(supervisor.GATE_STATE, {}) or {}
    gate = gate_state.get("gates", {}).get("reference_resolution", {})
    registered = {
        (str(item.get("path", "")).replace("\\", "/"), str(item.get("sha256", "")).upper())
        for item in gate.get("evidence", [])
    }
    expected = (relative_path(source_path), file_digest(source_path))
    if gate.get("passed") is not True or expected not in registered:
        raise ValueError("approved protocol is not bound to the registered reference gate")


def geometry_manifest_hash(params: list[tuple[float, float, float, float]]) -> str:
    array = np.asarray(params, dtype=np.float64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest().upper()


def canonicalize_params(
    params: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Put the long axis on x before solving both polarization channels."""
    return [
        (max(float(L), float(W)), min(float(L), float(W)), float(H), float(P))
        for L, W, H, P in params
    ]


def validate_pool_spec(
    spec: dict[str, Any], protocol: dict[str, Any], policy: dict[str, Any]
) -> tuple[Path, np.ndarray]:
    output = canonical_workspace_path(str(spec.get("path", "")), require_replacement_dir=True)
    forbidden = protected_paths(policy) | {canonical_workspace_path(policy["pool"]["path"])}
    if output in forbidden:
        raise ValueError("replacement output aliases a protected or historical asset")
    required = set(spec.get("required_record_fields", []))
    if not BASE_REQUIRED_FIELDS | REPLACEMENT_REQUIRED_FIELDS <= required:
        raise ValueError("replacement pool required fields are incomplete")
    if int(spec.get("expected_records", 0)) != int(protocol["samples"]) * 2:
        raise ValueError("replacement expected_records must equal samples times two polarizations")
    if list(spec.get("polarizations", [])) != ["p", "s"]:
        raise ValueError("replacement pool requires ordered p/s polarization pairs")
    for key in ("material", "substrate"):
        if spec.get(key) != protocol.get(key):
            raise ValueError(f"pool spec {key} differs from approved protocol")
    if int(spec.get("nG_requested", -1)) != int(protocol["nG_requested"]):
        raise ValueError("pool spec nG differs from approved protocol")
    baseline = policy["pool"]
    for name in (
        "range_tolerance", "pointwise_conservation_tolerance",
        "stored_value_tolerance", "quality_tolerance",
    ):
        if float(spec.get(name, float("inf"))) > float(baseline[name]):
            raise ValueError(f"replacement pool weakens {name}")
    if spec.get("lossless") is not True or protocol.get("background") != "air":
        raise ValueError("replacement protocol must remain lossless with air background")
    wavelength = colorimetry.wavelength_grid(float(protocol["wavelength_step_nm"]))
    if list(map(float, spec.get("wavelength_nm", []))) != wavelength.tolist():
        raise ValueError("pool spec wavelength grid differs from approved protocol")
    expected_meta = spec.get("expected_meta", {})
    exact_meta = {
        "seed": int(protocol["seed"]),
        "nG": int(protocol["nG_requested"]),
        "Nxy": int(protocol["Nxy"]),
        "material": protocol["material"],
        "substrate": protocol["substrate"],
        "background": protocol["background"],
        "pols": ["p", "s"],
        "n_samples": int(protocol["samples"]),
        "sampler_version": protocol["sampler_version"],
        "quality_rule": protocol["quality_rule"],
        "wavelength_step_nm": float(protocol["wavelength_step_nm"]),
        "colorimetry_version": colorimetry.COLORIMETRY_VERSION,
        "axis_canonicalization": "L=max(raw_axes), W=min(raw_axes); recompute p/s on canonical axes",
    }
    if expected_meta != exact_meta:
        raise ValueError("pool expected_meta is not the exact approved protocol metadata")
    return output, wavelength


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("replacement protocol schema_version must be 1")
    if (
        protocol.get("evidence_version") != PROTOCOL_VERSION
        or protocol.get("protocol_revision") != "v2_bound_holdout"
        or protocol.get("approved") is not True
    ):
        raise ValueError("replacement protocol is not auditor-approved")
    integrity = supervisor.verify_policy_integrity(supervisor.load_policy())
    if integrity.get("passed") is not True:
        raise ValueError("pipeline policy integrity is not verified")
    source_gate_matches(protocol)
    policy = supervisor.load_policy()
    baseline_meta = policy["pool"]["expected_meta"]
    if int(protocol.get("samples", -1)) * 2 != int(policy["pool"]["expected_records"]):
        raise ValueError("replacement protocol cannot reduce the approved pool size")
    for key in ("seed", "material", "substrate", "sampler_version"):
        if protocol.get(key) != baseline_meta.get(key):
            raise ValueError(f"replacement protocol changes frozen field {key}")
    if int(protocol.get("Nxy", 0)) < 64 or int(protocol.get("nG_requested", 0)) < 1:
        raise ValueError("replacement numerical budget is invalid")
    runtime_paths = (
        "rcwa_batch.py",
        "paper2_colorimetry_fine.py",
        "scripts/run_replacement_pool.py",
    )
    expected_runtime = protocol.get("runtime_hashes")
    actual_runtime = {name: file_digest(ROOT / name) for name in runtime_paths}
    if expected_runtime != actual_runtime:
        raise ValueError("replacement protocol runtime hashes do not match disk")
    output, wavelength = validate_pool_spec(protocol.get("pool_spec", {}), protocol, policy)
    if int(protocol.get("max_same_config_attempts", 0)) not in range(1, 4):
        raise ValueError("max_same_config_attempts must be between one and three")
    params = canonicalize_params(
        generate_params_elliptical(int(protocol["samples"]), seed=int(protocol["seed"]))
    )
    if geometry_manifest_hash(params) != str(protocol.get("geometry_manifest_sha256", "")).upper():
        raise ValueError("approved geometry manifest hash mismatch")
    retained = {
        retained_order(int(protocol["nG_requested"]), float(values[3])) for values in params
    }
    if retained != {int(protocol.get("nG_retained", -1))}:
        raise ValueError("approved retained order does not match grcwa truncation")
    return {
        "protocol": protocol,
        "protocol_path": path,
        "protocol_sha256": file_digest(path),
        "policy": policy,
        "output": output,
        "wavelength": wavelength,
        "params": params,
    }


def geometry_id(values: tuple[float, float, float, float]) -> str:
    encoded = "|".join(f"{float(value):.17g}" for value in values).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:24]


def build_tasks(context: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = context["protocol"]
    tasks: list[dict[str, Any]] = []
    ordinal = 0
    for values in context["params"]:
        values = tuple(float(value) for value in values)
        identifier = geometry_id(values)
        for pol in ("p", "s"):
            tasks.append(
                {
                    "id": f"{identifier}-{pol}",
                    "ordinal": ordinal,
                    "geometry_id": identifier,
                    "geometry": values,
                    "pol": pol,
                    "wavelength_nm": context["wavelength"],
                    "nG_requested": int(protocol["nG_requested"]),
                    "nG_retained": int(protocol["nG_retained"]),
                    "Nxy": int(protocol["Nxy"]),
                    "material": protocol["material"],
                    "substrate": protocol["substrate"],
                    "background": protocol["background"],
                    "quality_tolerance": float(context["protocol"]["pool_spec"]["quality_tolerance"]),
                    "conservation_tolerance": float(
                        context["protocol"]["pool_spec"]["pointwise_conservation_tolerance"]
                    ),
                    "max_attempts": int(protocol["max_same_config_attempts"]),
                    "wavelength_step_nm": float(protocol["wavelength_step_nm"]),
                }
            )
            ordinal += 1
    return tasks


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    L, W, H, P = task["geometry"]
    started = time.perf_counter()
    errors: list[str] = []
    for attempt in range(1, task["max_attempts"] + 1):
        try:
            reflectance, transmittance = rcwa_spectrum(
                L,
                H,
                P,
                task["wavelength_nm"],
                nG_req=task["nG_requested"],
                Nxy=task["Nxy"],
                material=task["material"],
                substrate=task["substrate"],
                W_nm=W,
                pol=task["pol"],
                background=task["background"],
            )
            R = np.asarray(reflectance, dtype=np.float64)
            T = np.asarray(transmittance, dtype=np.float64)
            if R.shape != task["wavelength_nm"].shape or T.shape != R.shape:
                raise ValueError("solver returned an unexpected spectrum shape")
            if not np.isfinite(R).all() or not np.isfinite(T).all():
                raise ValueError("solver returned non-finite spectra")
            pointwise_error = float(np.max(np.abs(R + T - 1.0)))
            if pointwise_error > task["conservation_tolerance"]:
                raise ValueError(f"pointwise conservation error {pointwise_error:.6g}")
            labels = colorimetry.spectrum_to_labels_d65(R, task["wavelength_nm"])
            rt_mean = float(np.mean(R + T))
            record = {
                "geometry_id": task["geometry_id"],
                "L": L,
                "W": W,
                "H": H,
                "P": P,
                "r": max(L, W) / min(L, W),
                "pol": task["pol"],
                "material": task["material"],
                "substrate": task["substrate"],
                "background": task["background"],
                "nG_requested": task["nG_requested"],
                "nG_retained": task["nG_retained"],
                "nG_actual": task["nG_requested"],
                "retry_nG": task["nG_requested"],
                "Nxy": task["Nxy"],
                "isolated": False,
                "wavelength_step_nm": task["wavelength_step_nm"],
                "wl_nm": np.asarray(task["wavelength_nm"], dtype=np.float64),
                "R": R,
                "T": T,
                "R_plus_T_mean": rt_mean,
                "pointwise_conservation_error_max": pointwise_error,
                "quality_pass": abs(rt_mean - 1.0) <= task["quality_tolerance"],
                "xyz": np.asarray(labels["xyz"], dtype=np.float64),
                "lab": np.asarray(labels["lab"], dtype=np.float64),
                "srgb_display": np.asarray(labels["srgb_display"], dtype=np.float64),
                "label_provenance_version": colorimetry.COLORIMETRY_VERSION,
                "attempts": attempt,
                "time_s": time.perf_counter() - started,
                "success": True,
            }
            return {"id": task["id"], "ordinal": task["ordinal"], "status": "ok", "record": record}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "id": task["id"],
        "ordinal": task["ordinal"],
        "status": "failed",
        "errors": errors,
        "task": {key: task[key] for key in ("geometry_id", "geometry", "pol", "nG_requested", "nG_retained", "Nxy", "wavelength_step_nm")},
        "time_s": time.perf_counter() - started,
    }


class Checkpoint:
    def __init__(self, path: Path, identity: dict[str, str], resume: bool):
        self.path = path
        if path.exists() and not resume:
            raise ValueError("checkpoint exists; use --resume after verifying the approved protocol")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS results (task_id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL, status TEXT NOT NULL, payload BLOB NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS failure_log (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, observed_at TEXT NOT NULL, payload BLOB NOT NULL)"
        )
        existing = {
            key: value for key, value in self.connection.execute("SELECT key, value FROM metadata")
        }
        encoded = {key: json.dumps(value, sort_keys=True) for key, value in identity.items()}
        if existing and existing != encoded:
            self.connection.close()
            raise ValueError("checkpoint identity differs from the approved protocol")
        if not existing:
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", encoded.items()
            )
            self.connection.commit()

    def successful_ids(self) -> set[str]:
        return {
            row[0]
            for row in self.connection.execute("SELECT task_id FROM results WHERE status='ok'")
        }

    def store(self, result: dict[str, Any]) -> None:
        payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        with self.connection:
            self.connection.execute(
                "INSERT INTO results(task_id, ordinal, status, payload) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(task_id) DO UPDATE SET ordinal=excluded.ordinal, status=excluded.status, payload=excluded.payload",
                (result["id"], int(result["ordinal"]), result["status"], payload),
            )
            if result["status"] != "ok":
                self.connection.execute(
                    "INSERT INTO failure_log(task_id, observed_at, payload) VALUES (?, ?, ?)",
                    (result["id"], supervisor.now_iso(), payload),
                )

    def results(self) -> list[dict[str, Any]]:
        return [
            pickle.loads(row[0])
            for row in self.connection.execute("SELECT payload FROM results ORDER BY ordinal")
        ]

    def failure_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM failure_log").fetchone()[0])

    def close(self) -> None:
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def task_manifest_hash(tasks: list[dict[str, Any]]) -> str:
    manifest = [
        {
            "id": task["id"],
            "ordinal": task["ordinal"],
            "geometry": task["geometry"],
            "pol": task["pol"],
            "nG_requested": task["nG_requested"],
            "nG_retained": task["nG_retained"],
            "Nxy": task["Nxy"],
            "wavelength_step_nm": task["wavelength_step_nm"],
        }
        for task in tasks
    ]
    encoded = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def build_pool_meta(context: dict[str, Any]) -> dict[str, Any]:
    protocol = context["protocol"]
    meta = dict(protocol["pool_spec"]["expected_meta"])
    meta.update(
        {
            "nG_requested": int(protocol["nG_requested"]),
            "nG_retained": int(protocol["nG_retained"]),
            "runner_version": RUNNER_VERSION,
            "approved_protocol": {
                "path": relative_path(context["protocol_path"]),
                "sha256": context["protocol_sha256"],
            },
            "label_provenance": colorimetry.label_provenance(
                float(protocol["wavelength_step_nm"])
            ),
        }
    )
    return meta


def publish_pool(
    context: dict[str, Any], results: list[dict[str, Any]], checkpoint_path: Path
) -> dict[str, Any]:
    expected = int(context["protocol"]["pool_spec"]["expected_records"])
    if len(results) != expected or any(item.get("status") != "ok" for item in results):
        raise RuntimeError("replacement pool is incomplete; final output will not be published")
    records = [item["record"] for item in results]
    output = context["output"]
    if output.exists():
        raise ValueError("final replacement output already exists; never overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    payload = {"meta": build_pool_meta(context), "records": records}
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    audit = supervisor.audit_pool(temporary, context["protocol"]["pool_spec"])
    if audit.get("passed") is not True:
        raise RuntimeError(f"strict replacement pool audit failed: {audit.get('errors', [])[:5]}")
    replace_with_retry(temporary, output)
    return audit


def build_evidence(
    context: dict[str, Any], audit: dict[str, Any], checkpoint_path: Path, failure_events: int
) -> dict[str, Any]:
    output = context["output"]
    source = context["protocol"]["source_reference_gate"]
    runtime_paths = (
        "rcwa_batch.py",
        "paper2_colorimetry_fine.py",
        "scripts/run_replacement_pool.py",
    )
    pool_sha256 = file_digest(output)
    return {
        "schema_version": 1,
        "evidence_version": EVIDENCE_VERSION,
        "passed": True,
        "activation_id": hashlib.sha256(
            f"{context['protocol_sha256']}|{pool_sha256}".encode("ascii")
        ).hexdigest()[:24],
        "pool_sha256": pool_sha256,
        "pool_md5": file_digest(output, "md5"),
        "size_bytes": output.stat().st_size,
        "pool_spec": context["protocol"]["pool_spec"],
        "approved_protocol": {
            "path": relative_path(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        },
        "reference_gate_evidence": source,
        "checkpoint": {
            "path": relative_path(checkpoint_path),
            "sha256": file_digest(checkpoint_path),
            "failure_events": failure_events,
        },
        "runtime_hashes": {path: file_digest(ROOT / path) for path in runtime_paths},
        "audit": {
            key: audit.get(key)
            for key in (
                "records", "expected_records", "geometries", "complete_pairs",
                "duplicate_keys", "R_plus_T_mean", "R_plus_T_min", "R_plus_T_max",
                "pointwise_conservation_error_max", "R_min", "R_max", "T_min", "T_max",
            )
        },
        "protected_files": supervisor.audit_protected_files(context["policy"]),
        "training_allowed": False,
        "generated_at": supervisor.now_iso(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = canonical_workspace_path(args.approved_protocol)
    context = validate_protocol(protocol_path)
    tasks = build_tasks(context)
    manifest_sha256 = task_manifest_hash(tasks)
    checkpoint_path = canonical_workspace_path(args.checkpoint)
    evidence_path = canonical_workspace_path(args.evidence)
    lock_path = canonical_workspace_path(args.lock)
    identity = {
        "runner_version": RUNNER_VERSION,
        "protocol_sha256": context["protocol_sha256"],
        "task_manifest_sha256": manifest_sha256,
    }
    if args.preflight:
        return {
            "preflight": True,
            "protocol_sha256": context["protocol_sha256"],
            "task_manifest_sha256": manifest_sha256,
            "tasks": len(tasks),
            "output": relative_path(context["output"]),
        }
    if context["output"].exists() or evidence_path.exists():
        raise ValueError("final output or evidence already exists; use a new versioned protocol")
    with RunLock(lock_path, context["protocol_sha256"]):
        checkpoint = Checkpoint(checkpoint_path, identity, args.resume)
        try:
            completed = checkpoint.successful_ids()
            pending = [task for task in tasks if task["id"] not in completed]
            with Pool(max(1, int(args.n_jobs))) as workers:
                for index, result in enumerate(
                    workers.imap_unordered(run_task, pending, chunksize=1), 1
                ):
                    checkpoint.store(result)
                    if index % max(1, int(args.progress_every)) == 0:
                        print(
                            json.dumps(
                                {
                                    "completed_this_run": index,
                                    "remaining": len(pending) - index,
                                    "status": result["status"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            results = checkpoint.results()
            failure_events = checkpoint.failure_count()
        finally:
            checkpoint.close()
        audit = publish_pool(context, results, checkpoint_path)
        evidence = build_evidence(context, audit, checkpoint_path, failure_events)
        atomic_json(evidence_path, evidence)
        return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-protocol", required=True)
    parser.add_argument("--checkpoint", default=".state/replacement_pool_v1.sqlite")
    parser.add_argument("--evidence", default=".state/replacement_pool_v1.json")
    parser.add_argument("--lock", default=".state/replacement_pool_v1.lock")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
