import numpy as np


def test_inverse_design_rcwa_reuses_cached_model(monkeypatch):
    import torch
    import torch_model

    class FakeModel:
        def __call__(self, x):
            return torch.full((x.shape[0], 81), 0.1, dtype=torch.float32)

    model = FakeModel()
    loads = []
    monkeypatch.setattr(
        torch_model, "_load_rcwa_torch_model",
        lambda device, material, substrate: loads.append(model) or model,
    )
    monkeypatch.setattr(torch_model.inverse_design_rcwa, "_cache", None, raising=False)

    first = torch_model.inverse_design_rcwa(
        np.array([0.4, 0.3, 0.2]), n_restarts=1, steps=0,
        p_fixed=400.0, device=torch.device("cpu"),
    )
    second = torch_model.inverse_design_rcwa(
        np.array([0.4, 0.3, 0.2]), n_restarts=1, steps=0,
        p_fixed=400.0, device=torch.device("cpu"),
    )

    assert loads == [model]
    assert first["pred_rgb"] == second["pred_rgb"]


def test_inverse_design_rcwa_invalidates_replaced_weights(monkeypatch, tmp_path):
    import torch
    import ml_module
    import torch_model

    artifact = tmp_path / 'checkpoint.pt'
    artifact.write_bytes(b'first')
    monkeypatch.setattr(ml_module, 'rcwa_torch_artifact_paths', lambda *_: (str(artifact),))
    loads = []

    class Model:
        def __call__(self, values):
            return torch.full((len(values), 81), 0.1)

    monkeypatch.setattr(torch_model, '_load_rcwa_torch_model', lambda *_: loads.append(1) or Model())
    monkeypatch.setattr(torch_model.inverse_design_rcwa, '_cache', None, raising=False)
    kwargs = dict(n_restarts=1, steps=0, device=torch.device('cpu'))
    torch_model.inverse_design_rcwa(torch.tensor([0.4, 0.3, 0.2]), **kwargs)
    artifact.write_bytes(b'replacement with new identity')
    torch_model.inverse_design_rcwa(torch.tensor([0.4, 0.3, 0.2]), **kwargs)
    assert len(loads) == 2


def test_unknown_rcwa_material_never_uses_tio2_weights():
    import pytest
    import torch_model
    with pytest.raises(ValueError, match='No registered'):
        torch_model._load_rcwa_torch_model(material='unknown material')
