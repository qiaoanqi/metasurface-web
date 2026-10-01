"""Generate a deterministic SHA-256 manifest for the UI release surface.

Only the fixed whitelist from :mod:`verify_ui_release` is read.  The default
output is stdout; ``--output`` is an explicit opt-in for writing the generated
manifest.  Source files are never changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from verify_ui_release import REQUIRED_FILES, _safe_path, _sha256  # noqa: E402


MANIFEST_SCHEMA = "ui-release-manifest-v1"
MANIFEST_FILES: tuple[str, ...] = REQUIRED_FILES


def build_manifest(
    root: Path,
    whitelist: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a stable JSON-compatible manifest without writing any source."""
    root = root.resolve()
    paths = tuple(MANIFEST_FILES if whitelist is None else whitelist)
    files: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    for relative in sorted(set(paths)):
        normalized = str(relative).replace("\\", "/")
        try:
            target = _safe_path(root, normalized)
            if not target.is_file():
                raise FileNotFoundError(normalized)
            files[normalized] = "sha256:" + _sha256(target)
        except Exception as exc:
            errors.append({
                "path": normalized,
                "reason": f"{type(exc).__name__}: {exc}",
            })
    return {
        "files": files,
        "schema": MANIFEST_SCHEMA,
        "status": "pass" if not errors else "fail",
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    parser.add_argument("--output", type=Path, help="optional manifest output path")
    parser.add_argument("--pretty", action="store_true", help="indent JSON for humans")
    args = parser.parse_args(argv)
    report = build_manifest(args.root.resolve())
    indent = 2 if args.pretty else None
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )
    if args.output:
        args.output.resolve().write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
