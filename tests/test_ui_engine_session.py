import numpy as np
import pytest

from ui_engine_session import (
    ENGINE_LIBRARY_KEY,
    ENGINE_OWNER_KEY,
    ENGINE_SESSION_CONTRACT_VERSION,
    ENGINE_SESSION_KEY,
    LibraryIdentity,
    bind_engine_library,
    configure_engine_far_field,
    engine_library_matches,
    get_session_engine,
    make_local_bound_engine,
)


class StubEngine:
    def __init__(self):
        self._last_material = "TiO2"
        self._last_substrate = "SiO2"
        self._last_polarization = "TE"
        self._last_angle = 0.0
        self._enable_far_field = False
        self._na = 0.1
        self._theta_obs_deg = 0.0
        self.rebuild_calls = []
        self._set_grid(1.0)

    def _set_grid(self, value):
        self.grid_params = np.array([[100.0 + value, 200.0, 300.0]])
        self.grid_rgb = np.array([[0.1 * value, 0.2, 0.3]])
        self.grid_lab = np.array([[10.0 * value, 20.0, 30.0]])
        self.grid_xy = np.array([[0.2, 0.3]])

    def rebuild_library(self, material, substrate, polarization, angle_deg):
        self._last_material = material
        self._last_substrate = substrate
        self._last_polarization = polarization
        self._last_angle = float(angle_deg)
        self.rebuild_calls.append((material, substrate, polarization, float(angle_deg)))
        self._set_grid(float(len(self.rebuild_calls) + 1))


class LazyStubEngine(StubEngine):
    def __init__(self):
        super().__init__()
        self.grid_params = np.zeros((0, 3))
        self.grid_rgb = np.zeros((0, 3))
        self.grid_lab = np.zeros((0, 3))
        self.grid_xy = np.zeros((0, 2))


def identity(
        material="TiO2", substrate="SiO2", polarization="TE", angle=0.0,
        far_field=False, na=0.1, theta=0.0):
    return LibraryIdentity(
        material, substrate, polarization, angle, far_field, na, theta)


def test_two_sessions_never_share_mutable_engine_or_far_field_state():
    state_a = {}
    state_b = {}
    engine_a = get_session_engine(state_a, StubEngine)
    engine_b = get_session_engine(state_b, StubEngine)

    assert engine_a is not engine_b
    assert state_a[ENGINE_OWNER_KEY] != state_b[ENGINE_OWNER_KEY]
    configure_engine_far_field(engine_a, True, 0.5, 20.0)
    assert (engine_a._enable_far_field, engine_a._na, engine_a._theta_obs_deg) == (
        True, 0.5, 20.0)
    assert (engine_b._enable_far_field, engine_b._na, engine_b._theta_obs_deg) == (
        False, 0.1, 0.0)

    configure_engine_far_field(engine_a, False, 0.9, 70.0)
    assert (engine_a._enable_far_field, engine_a._na, engine_a._theta_obs_deg) == (
        False, 0.1, 0.0)


def test_disabled_far_field_identity_canonicalizes_stale_controls():
    stale = identity(far_field=False, na=0.5, theta=20.0)
    engine = StubEngine()
    state = {}

    assert (stale.far_field_enabled, stale.na, stale.theta_obs_deg) == (
        False, 0.1, 0.0)
    bind_engine_library(engine, stale, state)
    assert engine_library_matches(engine, stale)
    assert state[ENGINE_LIBRARY_KEY] == stale
    assert (engine._enable_far_field, engine._na, engine._theta_obs_deg) == (
        False, 0.1, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"angle": -0.1},
        {"angle": 80.1},
        {"far_field": True, "na": 0.049},
        {"far_field": True, "na": 0.951},
        {"far_field": True, "theta": -0.1},
        {"far_field": True, "theta": 80.1},
        {"far_field": True, "na": np.nan},
    ],
)
def test_library_identity_rejects_nonfinite_or_out_of_range_values(kwargs):
    with pytest.raises(ValueError):
        identity(**kwargs)


def test_library_rebuilds_and_pattern_like_reads_are_session_local():
    state_a = {}
    state_b = {}
    engine_a = get_session_engine(state_a, StubEngine)
    engine_b = get_session_engine(state_b, StubEngine)
    identity_a = identity("a-Si", "Glass", "TM", 15.0, True, 0.5, 20.0)
    identity_b = identity("TiO2", "SiO2", "TE", 0.0)

    bind_engine_library(engine_b, identity_b, state_b)
    b_grid_before = engine_b.grid_rgb.copy()
    bind_engine_library(engine_a, identity_a, state_a)
    pattern_read_a = engine_a.grid_lab.copy()

    assert engine_library_matches(engine_a, identity_a)
    assert engine_library_matches(engine_b, identity_b)
    assert state_a[ENGINE_LIBRARY_KEY] == identity_a
    assert state_b[ENGINE_LIBRARY_KEY] == identity_b
    np.testing.assert_array_equal(engine_b.grid_rgb, b_grid_before)
    assert pattern_read_a is not engine_b.grid_lab

    bind_engine_library(engine_b, identity_b, state_b)
    np.testing.assert_array_equal(engine_b.grid_rgb, b_grid_before)
    assert engine_a._last_material == "a-Si"
    assert engine_b._last_material == "TiO2"


def test_lazy_engine_matching_identity_rebuilds_once_then_reuses_valid_grid():
    engine = LazyStubEngine()
    target = identity()

    bind_engine_library(engine, target)

    assert engine.rebuild_calls == [("TiO2", "SiO2", "TE", 0.0)]
    assert engine_library_matches(engine, target)
    for name, width in {
        "grid_params": 3, "grid_rgb": 3, "grid_lab": 3, "grid_xy": 2,
    }.items():
        value = np.asarray(getattr(engine, name))
        assert value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == width
        assert np.all(np.isfinite(value))

    bind_engine_library(engine, target)
    assert engine.rebuild_calls == [("TiO2", "SiO2", "TE", 0.0)]


def test_local_compare_engine_never_changes_main_session_identity():
    state = {}
    main_engine = get_session_engine(state, StubEngine)
    main_identity = identity("TiO2", "SiO2", "TE", 0.0)
    bind_engine_library(main_engine, main_identity, state)
    main_grid_before = main_engine.grid_rgb.copy()

    compare_engine = make_local_bound_engine(
        StubEngine, identity("TiO2", "Glass", "TE", 0.0))
    bind_engine_library(
        compare_engine, identity("a-Si", "Glass", "TE", 0.0))

    assert compare_engine is not main_engine
    assert compare_engine._last_material == "a-Si"
    assert engine_library_matches(main_engine, main_identity)
    np.testing.assert_array_equal(main_engine.grid_rgb, main_grid_before)


def test_hot_update_or_cross_owner_legacy_engine_is_rebuilt_fail_closed():
    legacy = StubEngine()
    state = {
        ENGINE_OWNER_KEY: "owner-a",
        ENGINE_SESSION_KEY: legacy,
        ENGINE_LIBRARY_KEY: identity(),
    }
    replacement = get_session_engine(state, StubEngine)

    assert replacement is not legacy
    assert state[ENGINE_SESSION_KEY] is replacement
    assert ENGINE_LIBRARY_KEY not in state
    assert replacement._ui_session_contract_version == ENGINE_SESSION_CONTRACT_VERSION
    assert replacement._ui_session_owner == "owner-a"

    state_other = {
        ENGINE_OWNER_KEY: "owner-b",
        ENGINE_SESSION_KEY: replacement,
    }
    other = get_session_engine(state_other, StubEngine)
    assert other is not replacement


def test_grid_identity_validation_fails_closed_on_unknown_or_corrupt_grid():
    class CorruptRebuildEngine(LazyStubEngine):
        def rebuild_library(self, material, substrate, polarization, angle_deg):
            self.rebuild_calls.append(
                (material, substrate, polarization, float(angle_deg)))

    engine = CorruptRebuildEngine()
    with pytest.raises(RuntimeError, match=r"grid_(params|rgb|lab|xy)"):
        bind_engine_library(engine, identity())
    assert engine.rebuild_calls == [("TiO2", "SiO2", "TE", 0.0)]
