import numpy as np
import pytest


def test_trained_negative_q_values_still_choose_greedy_action(monkeypatch):
    import rl_design as rl
    agent = rl.RLDesigner()
    agent.trained = True
    agent.q[:] = -2
    agent.q[:, 2] = -1  # H+5 is the best learned action, though negative.
    initial = iter([100., 200., 400.])
    monkeypatch.setattr(np.random, 'uniform', lambda *_: next(initial))
    monkeypatch.setattr(np.random, 'randint', lambda *_: pytest.fail('trained search used a random action'))
    geometries = []
    def predict(d, h, p, **_):
        geometries.append((d,h,p))
        return np.array([0.2, 0.3, 0.4])
    monkeypatch.setattr(rl, '_compute_rgb', predict)
    agent.search('#ffffff', steps=1, restarts=1)
    assert geometries == [(100.,200.,400.), (100.,205.,400.)]


def test_missing_qtable_never_trains_or_downloads(monkeypatch):
    import rl_design as rl
    monkeypatch.setattr(rl.RLDesigner, 'load', lambda *_, **__: False)
    monkeypatch.setattr(rl.RLDesigner, 'train', lambda *_: pytest.fail('unapproved training'))
    monkeypatch.setattr(rl, '_ensure_model_file', lambda *_: pytest.fail('unexpected download'))
    with pytest.raises(FileNotFoundError, match='never trains'):
        rl.get_trained_rl()
