#!/usr/bin/env bash
# Prepare a complete release while the existing website stays online, then
# switch the application root and restart only its systemd service.
set -Eeuo pipefail

ZIP="${1:?ZIP path required}"
EXPECTED="${2:?SHA-256 required}"
RELEASE_ID="${3:?release id required}"
INSTALL_ROOT=/opt/ai-metasurface-web
SERVICE=metasurface-streamlit
[[ ${EUID} -eq 0 ]]
[[ "$ZIP" == /root/*.zip && -f "$ZIP" ]]
[[ "$EXPECTED" =~ ^[A-Fa-f0-9]{64}$ ]]
[[ "$RELEASE_ID" =~ ^[A-Za-z0-9_-]+$ ]]
RELEASE_ROOT="/opt/ai-metasurface-web.release.${RELEASE_ID}"
BACKUP_ROOT="/opt/ai-metasurface-web.backup.$(date +%Y%m%d%H%M%S)"
test ! -e "$RELEASE_ROOT"
test ! -e "$BACKUP_ROOT"
printf '%s  %s\n' "$EXPECTED" "$ZIP" | sha256sum -c -
for binary in python3 curl nginx systemctl; do command -v "$binary" >/dev/null; done
test -d "$INSTALL_ROOT/.venv"

mkdir -m 0755 "$RELEASE_ROOT"
python3 - "$ZIP" "$RELEASE_ROOT" <<'PY'
import hashlib, json, pathlib, stat, sys, zipfile
archive_path, destination = sys.argv[1], pathlib.Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive_path) as archive:
    manifest = json.loads(archive.read('RELEASE_MANIFEST.json'))
    expected_files = set(manifest['files']) | {'RELEASE_MANIFEST.json'}
    assert set(archive.namelist()) == expected_files
    assert len(archive.namelist()) == len(expected_files)
    for info in archive.infolist():
        target = (destination / info.filename).resolve()
        target.relative_to(destination)
        assert not stat.S_ISLNK(info.external_attr >> 16)
        if info.filename != 'RELEASE_MANIFEST.json':
            digest = 'sha256:' + hashlib.sha256(archive.read(info)).hexdigest().upper()
            assert digest == manifest['files'][info.filename], info.filename
    archive.extractall(destination)
print('RELEASE_FILES_VERIFIED=' + str(len(manifest['files'])))
PY

# Reuse installed dependencies in a private copy. The running release is not
# modified, and each release keeps its own virtual environment for rollback.
cp -a --reflink=auto "$INSTALL_ROOT/.venv" "$RELEASE_ROOT/.venv"
python3 -m venv --upgrade "$RELEASE_ROOT/.venv"
PYTHON="$RELEASE_ROOT/.venv/bin/python"
if ! "$PYTHON" -c 'import torch; assert torch.version.cuda is None' >/dev/null 2>&1; then
    "$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2,<3'
fi
"$PYTHON" -m pip install -r "$RELEASE_ROOT/requirements-web.txt"
cd "$RELEASE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=-1
"$PYTHON" - <<'PY'
from pathlib import Path
from scripts.verify_complete_offline_bundle import validate_analysis_runtime
import ml_module
from competition.reference_library import load_reference_library
report = validate_analysis_runtime(Path('.'))
assert ml_module.init_ml() and ml_module.init_rcwa_ml()
library = load_reference_library()
assert library.record_count == 21088 and library.lookup(140, 281, 407) is not None
print('ASSET_PREFLIGHT_OK dependencies=' + str(len(report['files'])))
PY
"$PYTHON" scripts/audit_offline_app_flows.py --runtime "$RELEASE_ROOT" --output "$RELEASE_ROOT/deployment-app-audit.json"
install -d -m 0750 -o metasurface -g metasurface "$RELEASE_ROOT/.runtime-home"
chown -R metasurface:metasurface "$RELEASE_ROOT"
nginx -t

switched=0
rollback() {
    local rc=$?
    if [[ "$switched" == 1 ]]; then
        mv "$INSTALL_ROOT" "${RELEASE_ROOT}.failed-link"
        mv "$BACKUP_ROOT" "$INSTALL_ROOT"
        systemctl restart "$SERVICE"
        echo 'ROLLED_BACK_TO_PREVIOUS_RELEASE'
    fi
    exit "$rc"
}
trap rollback ERR
systemctl stop "$SERVICE"
mv "$INSTALL_ROOT" "$BACKUP_ROOT"
ln -s "$RELEASE_ROOT" "$INSTALL_ROOT"
switched=1
install -m 0644 "$RELEASE_ROOT/deployment/systemd/metasurface-streamlit.service" "/etc/systemd/system/${SERVICE}.service"
systemctl daemon-reload
systemctl start "$SERVICE"
for attempt in {1..30}; do
    if curl -fsS --max-time 3 http://127.0.0.1/app/_stcore/health >/dev/null; then break; fi
    sleep 1
done
bash "$RELEASE_ROOT/deployment/scripts/healthcheck.sh" http://127.0.0.1
systemctl is-active --quiet "$SERVICE"
systemctl is-active --quiet nginx
switched=0
trap - ERR
printf 'DEPLOYMENT_SUCCESS\nRELEASE_ROOT=%s\nBACKUP_ROOT=%s\n' "$RELEASE_ROOT" "$BACKUP_ROOT"
sha256sum RELEASE_MANIFEST.json app.py ml_module.py static/index.html competition/tio2_air_reference_records_v1.jsonl
systemctl show "$SERVICE" -p MainPID -p ActiveState -p ExecMainStartTimestamp
