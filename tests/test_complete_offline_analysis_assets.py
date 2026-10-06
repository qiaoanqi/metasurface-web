"""Release regression: manifests must not hide missing analysis evidence."""
from pathlib import Path
import shutil

import pytest

from scripts.verify_complete_offline_bundle import validate_analysis_runtime


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = 'scripts/audit_forward_mlp_v8_sub_conversion.py'


@pytest.fixture
def runtime(tmp_path):
    report = validate_analysis_runtime(ROOT)
    for relative in report['files']:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def test_analysis_dependency_inventory_includes_bound_auditor(runtime):
    report = validate_analysis_runtime(runtime)
    assert report['status'] == 'pass'
    assert AUDITOR in report['files']
    assert 'models/forward_mlp_v8_sub.onnx.data' in report['files']


@pytest.mark.parametrize('relative', [AUDITOR, 'ui_model_resources.py', 'data/fano_vs_fdtd_smallD.png', 'models/forward_mlp_rcwa_TiO2_s1.pt', 'models/rl_qtable.npy'])
def test_release_rejects_missing_analysis_dependency(runtime, relative):
    (runtime / relative).unlink()
    with pytest.raises(ValueError, match='Missing analysis dependency'):
        validate_analysis_runtime(runtime)


def test_release_rejects_altered_conversion_auditor(runtime):
    with (runtime / AUDITOR).open('ab') as handle:
        handle.write(b'\n# unexpected edit\n')
    with pytest.raises(ValueError, match='audit script hash mismatch'):
        validate_analysis_runtime(runtime)
