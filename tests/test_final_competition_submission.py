import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "submission_builder", Path(__file__).resolve().parents[1] / "scripts" / "build_final_competition_submission.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def make_video(tmp_path):
    video = tmp_path / "demo_v5.mp4"
    video.write_bytes(b"verified-video-v5")
    receipt = {
        "status": "verified", "filename": video.name, "bytes": video.stat().st_size,
        "sha256": BUILDER.sha256(video).lower(), "fullDecode": {"exitCode": 0},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return video, receipt_path, receipt


def test_verified_video_accepts_lowercase_receipt_hash(tmp_path):
    video, receipt_path, receipt = make_video(tmp_path)
    assert BUILDER.verify_video_artifact(video, receipt_path) == receipt


@pytest.mark.parametrize("field,value", [
    ("status", "pending"), ("filename", "demo_v4.mp4"), ("bytes", 1),
    ("sha256", "0" * 64), ("fullDecode", {"exitCode": 1}),
])
def test_video_receipt_mismatch_fails_closed(tmp_path, field, value):
    video, receipt_path, receipt = make_video(tmp_path)
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        BUILDER.verify_video_artifact(video, receipt_path)


@pytest.mark.parametrize("value", ["v0", "v-1", "v29/../v4", "29"])
def test_package_version_cannot_escape_output_directory(value):
    with pytest.raises(argparse.ArgumentTypeError):
        BUILDER.version_arg(value)


def test_default_video_paths_select_the_fixed_version(monkeypatch):
    monkeypatch.setattr(BUILDER, "VIDEO_VERSION", "v5")
    video, receipt = BUILDER.video_sources()
    assert video.name.endswith("_v5.mp4")
    assert receipt.name.endswith("_v5_verification-receipt.json")
    assert BUILDER.video_destinations()[0].endswith("_v5.mp4")


@pytest.mark.parametrize("tamper", [None, "receipt", "stale_video", "root", "runtime"])
def test_submission_archive_checks_embedded_video_and_inventory(tmp_path, tamper):
    video, receipt_path, receipt = make_video(tmp_path)
    video_path, receipt_relative = BUILDER.video_destinations()
    runtime_root = "04_software/offline"
    launcher = runtime_root + "/start.bat"
    launcher_bytes = b"@echo off\r\n"
    launcher_hash = "sha256:" + hashlib.sha256(launcher_bytes).hexdigest().upper()
    runtime_manifest = {
        "status": "pass", "file_count": 1,
        "files": {"start.bat": "sha256:" + "0" * 64 if tamper == "runtime" else launcher_hash},
    }
    if tamper == "receipt":
        receipt["sha256"] = "0" * 64
    contents = {
        video_path: video.read_bytes(), receipt_relative: json.dumps(receipt).encode(),
        launcher: launcher_bytes,
        runtime_root + "/RELEASE_MANIFEST.json": json.dumps(runtime_manifest).encode(),
    }
    if tamper == "stale_video":
        contents[video_path.replace("v5.mp4", "v4.mp4")] = b"old-video"
    hashes = {name: "sha256:" + hashlib.sha256(data).hexdigest().upper() for name, data in contents.items()}
    manifest = {
        "status": "pass", "files": hashes, "file_count": len(hashes),
        "runtime_release_manifest_sha256": hashes[runtime_root + "/RELEASE_MANIFEST.json"].split(":")[1],
        "required_items": {"demo_video": video_path, "demo_video_receipt": receipt_relative, "offline_entrypoint": launcher},
        "video_delivery": {
            "version": "v5", "path": video_path, "receipt": receipt_relative,
            "source_filename": video.name, "sha256": BUILDER.sha256(video),
            "receipt_sha256": hashes[receipt_relative].split(":")[1],
        },
    }
    archive_path = tmp_path / "submission.zip"
    with ZipFile(archive_path, "w") as archive:
        for name, data in contents.items():
            archive.writestr("v29/" + name, data)
        archive.writestr(("v28/" if tamper == "root" else "v29/") + "SUBMISSION_MANIFEST.json", json.dumps(manifest))
    if tamper:
        with pytest.raises(ValueError):
            BUILDER.verify_submission_archive(archive_path)
    else:
        result = BUILDER.verify_submission_archive(archive_path)
        assert result["status"] == "pass"
        assert result["video_version"] == "v5"
        assert result["files_verified"] == len(hashes)
