"""Build the competition web deployment bundle without research/control-plane files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

if __package__:
    from .verify_complete_offline_bundle import validate_analysis_runtime
else:
    from verify_complete_offline_bundle import validate_analysis_runtime


CORE_FILES = (
    "app.py", "engine.py", "ccm.py", "color_utils.py", "fp_cavity.py",
    "ml_module.py", "rl_design.py", "torch_model.py", "requirements-web.txt",
    "ui_analysis_snapshots.py", "ui_benchmark_contracts.py", "ui_cie_contracts.py",
    "ui_engine_session.py", "ui_fdtd_asset.py", "ui_forward_routes.py",
    "ui_inverse_contracts.py", "ui_model_difference_contracts.py",
    "ui_model_resources.py", "ui_pattern_contracts.py", "ui_session_migration.py",
    ".streamlit/config.toml",
    "protocols/forward_mlp_v8_sub_conversion_v1.json",
    "models/evidence/forward_mlp_v8_sub_conversion_v1.json",
    "models/forward_mlp_v8_sub.onnx", "models/forward_mlp_v8_sub.onnx.data",
    "models/forward_mlp_v8_sub.pt", "models/rl_qtable.npy", "models/rl_qtable_meta.npy",
    "scripts/audit_offline_feature_runtime.py",
    "scripts/audit_offline_app_flows.py",
    "scripts/verify_complete_offline_bundle.py",
)

COMPETITION_FILES = (
    "competition/reference_library.py",
    "competition/tio2_air_reference_records_v1.jsonl",
    "competition/tio2_air_day_audit_20260930.json",
    "competition/tio2_air_day_color_audit_20260930.json",
    "competition/11_本地离线演示说明.md",
    "competition/12_校赛本地交付清单.md",
    "competition/13_公开资源索引.md",
)

MODEL_GLOB = "models/forward_mlp_rcwa_*.onnx"
STATIC_ROOT = Path("competition/website展示页_校赛副本")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def copy_file(root: Path, staging: Path, relative: str) -> None:
    source = root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="metasurface-web-") as temp_dir:
        staging = Path(temp_dir) / "ai_metasurface_web_bundle_v1"
        staging.mkdir()
        for relative in CORE_FILES + COMPETITION_FILES:
            copy_file(root, staging, relative)
        for source in sorted(root.glob(MODEL_GLOB)):
            copy_file(root, staging, source.relative_to(root).as_posix())
        # Use the same checked dependency inventory as the complete offline app.
        # It includes registered gradient weights, the bound conversion auditor,
        # and historical figure assets; no research datasets are copied.
        analysis_runtime = validate_analysis_runtime(root)
        for relative in analysis_runtime["files"]:
            copy_file(root, staging, relative)
        validate_analysis_runtime(staging)

        static_target = staging / "static"
        shutil.copytree(
            root / STATIC_ROOT,
            static_target,
            ignore=shutil.ignore_patterns("*.zip", "*.pptx"),
        )
        for relative in (
            "deployment/README.md", "deployment/install.sh",
            "deployment/nginx/metasurface.conf",
            "deployment/systemd/metasurface-streamlit.service",
            "deployment/scripts/healthcheck.sh",
            "deployment/scripts/update_release.sh",
        ):
            copy_file(root, staging, relative)

        files = {
            path.relative_to(staging).as_posix(): "sha256:" + sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema": "ai-metasurface-web-bundle-v1",
            "status": "pass",
            "analysis_runtime": analysis_runtime,
            "entrypoints": {"showcase": "/", "streamlit": "/app/"},
            "reference_records_sha256": files.get(
                "competition/tio2_air_reference_records_v1.jsonl", ""
            ).removeprefix("sha256:"),
            "files": files,
        }
        (staging / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(staging).as_posix()
                    info = ZipInfo(relative, date_time=(2020, 1, 1, 0, 0, 0))
                    info.compress_type = ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o644 << 16
                    archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "dist" / "ai_metasurface_web_bundle_20261011_v5.zip",
    )
    args = parser.parse_args()
    manifest = build(args.root, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "bytes": args.output.resolve().stat().st_size,
        "files": len(manifest["files"]),
        "reference_records_sha256": manifest["reference_records_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    print(f"sha256={sha256(args.output.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
