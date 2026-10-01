"""Extract a web bundle, start Streamlit, check its health endpoint, and stop it."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from zipfile import ZipFile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--port", type=int, default=8514)
    parser.add_argument(
        "--exercise-assets",
        action="store_true",
        help="after health check, initialize packaged models and exact reference lookup",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="metasurface-web-smoke-") as directory:
        root = Path(directory)
        with ZipFile(args.bundle) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP integrity test failed")
            archive.extractall(root)
        log_path = root / "streamlit.log"
        err_path = root / "streamlit.err.log"
        py_compile_files = [
            root / name for name in
            ("app.py", "engine.py", "ccm.py", "color_utils.py", "fp_cavity.py",
             "ml_module.py", "rl_design.py", "torch_model.py")
        ]
        subprocess.run(
            [sys.executable, "-m", "py_compile", *map(str, py_compile_files)],
            check=True,
        )
        command = [
            sys.executable, "-m", "streamlit", "run", str(root / "app.py"),
            "--server.address=127.0.0.1", f"--server.port={args.port}",
            "--server.baseUrlPath=app", "--server.headless=true",
            "--server.enableCORS=false", "--server.enableXsrfProtection=false",
            "--browser.gatherUsageStats=false",
        ]
        with log_path.open("w", encoding="utf-8") as log, err_path.open("w", encoding="utf-8") as err:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=err)
        health_url = f"http://127.0.0.1:{args.port}/app/_stcore/health"
        healthy = False
        try:
            for _ in range(60):
                time.sleep(0.5)
                try:
                    with urlopen(health_url, timeout=2) as response:
                        if response.status == 200 and response.read().strip() == b"ok":
                            healthy = True
                            break
                except Exception:
                    if process.poll() is not None:
                        break
            print({"pid": process.pid, "health_url": health_url, "healthy": healthy})
            if not healthy:
                print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
                print(err_path.read_text(encoding="utf-8", errors="replace")[-4000:])
                return 1
            if args.exercise_assets:
                sys.path.insert(0, str(root))
                ml_module = importlib.import_module("ml_module")
                ml_ready = bool(ml_module.init_ml())
                rcwa_ready = bool(ml_module.init_rcwa_ml())
                reference_module = importlib.import_module("competition.reference_library")
                library = reference_module.load_reference_library()
                match = library.lookup(140, 281, 407)
                asset_probe = {
                    "init_ml": ml_ready,
                    "init_rcwa_ml": rcwa_ready,
                    "reference_records": library.record_count,
                    "reference_unique_geometries": library.unique_geometry_count,
                    "exact_lookup_140_281_407": match is not None,
                }
                print(json.dumps({"asset_probe": asset_probe}, ensure_ascii=False, sort_keys=True))
                if not all((ml_ready, rcwa_ready, match is not None)):
                    return 1
            return 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
