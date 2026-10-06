"""Exercise deployment batch contracts with real ONNX Runtime inference."""
from pathlib import Path

import numpy as np
import pytest

import ml_module as ml
from color_utils import delta_e2000, rgb_to_lab

ort = pytest.importorskip('onnxruntime')
onnx = pytest.importorskip('onnx')


def _session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model, options, providers=['CPUExecutionProvider'])


def _identity_session(batch):
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node('Identity', ['params'], ['spectrum'])], 'batch-contract',
        [helper.make_tensor_value_info('params', TensorProto.FLOAT, [batch, 7])],
        [helper.make_tensor_value_info('spectrum', TensorProto.FLOAT, [batch, 7])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    return _session(model.SerializeToString())


@pytest.mark.parametrize('batch', [1, 4, 'batch', None])
def test_fixed_and_dynamic_models_preserve_every_input_row(batch):
    inputs = np.arange(1031 * 7, dtype=np.float32).reshape(1031, 7)
    output = ml._predict_onnx_batch(_identity_session(batch), inputs, 7)
    np.testing.assert_array_equal(output, inputs)


def test_batch_rejects_nonfinite_and_bad_output_shape():
    session = _identity_session(1)
    with pytest.raises(ValueError, match='non-finite'):
        ml._predict_onnx_batch(session, [[np.nan] * 7], 7)
    with pytest.raises(ValueError, match='output shape'):
        ml._predict_onnx_batch(session, np.ones((2, 7)), 81)
    with pytest.raises(ValueError, match='feature count'):
        ml._predict_onnx_batch(session, np.ones((2, 6)), 7)


def test_wavelength_route_also_supports_fixed_one_row_export():
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node('ReduceMean', ['params'], ['R'], axes=[1], keepdims=1)],
        'wavelength-contract',
        [helper.make_tensor_value_info('params', TensorProto.FLOAT, [1, 7])],
        [helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    output = ml._predict_rcwa_wavelength_with_session(
        _session(model.SerializeToString()), 'TiO2 (anatase)', 'SiO2 (fused silica)', 180, 300, 400)
    assert output.shape == (81,)
    assert np.all(np.isfinite(output))
    assert output[0] != output[-1]


@pytest.fixture
def deployed_ensemble(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'models' / f'forward_mlp_rcwa_TiO2_s{i}.onnx' for i in range(1, 4)]
    if not all(path.is_file() for path in paths):
        pytest.skip('TiO2 deployment artifacts are not available in this checkout')
    sessions = [_session(str(path)) for path in paths]
    assert all(session.get_inputs()[0].shape == [1, 7] for session in sessions)
    monkeypatch.setattr(ml, '_RCWA_SESSIONS', {
        ('TiO2 (anatase)', 'SiO2 (fused silica)'): sessions,
        'TiO2 (anatase)': sessions})
    monkeypatch.setattr(ml, '_RCWA_AVAILABLE', True)
    # No .pt weights, as in the actual RCWA release route.
    monkeypatch.setattr(ml, '__file__', str(tmp_path / 'ml_module.py'))
    return sessions


def test_deployed_ensemble_batch_matches_individual_predictions(deployed_ensemble):
    inputs = np.random.default_rng(20261006).random((17, 7), dtype=np.float32)
    for session in deployed_ensemble:
        batch = ml._predict_onnx_batch(session, inputs, 81)
        single = np.concatenate([session.run(None, {'params': row[None]})[0] for row in inputs])
        np.testing.assert_array_equal(batch, single)


def test_full_smart_grid_returns_three_finite_candidates(deployed_ensemble):
    target = np.array([0.25, 0.4, 0.7])
    candidates = ml.smart_grid_search(target)  # Real coarse grid + refinement.
    assert len(candidates) == 3
    for _, param, rgb, de76, de00 in candidates:
        assert param.period_nm >= param.diameter_nm * 1.2 - 0.1
        assert np.all(np.isfinite([*rgb, de76, de00]))
        assert np.all((np.asarray(rgb) >= 0) & (np.asarray(rgb) <= 1))
        assert de00 == pytest.approx(delta_e2000(rgb_to_lab(rgb), rgb_to_lab(target)))
        assert de76 == pytest.approx(np.linalg.norm(rgb_to_lab(rgb) - rgb_to_lab(target)))
