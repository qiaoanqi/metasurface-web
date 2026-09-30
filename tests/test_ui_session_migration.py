import math

import pytest

from ui_session_migration import (
    BoolControlSpec, EnumControlSpec, NumericControlSpec,
    ML_ACCEL_PREFERENCE_INITIALIZED_KEY,
    SESSION_MIGRATION_KEY, SESSION_MIGRATION_SCHEMA, SESSION_MIGRATION_VERSION,
    initialize_bool_preference_marker,
    migrate_session_state, normalize_bool, normalize_enum, normalize_numeric,
    set_bool_value, set_numeric_value, sync_bool_from_widget, sync_enum_from_widget,
    sync_bool_preference_from_widget, sync_numeric_from_widget,
)


NUMERIC = NumericControlSpec(
    "d_val", ("d_slider", "d_input"), 50.0, 350.0, 180.0)
ENUM = EnumControlSpec(
    "structure", "structure_widget", ("single", "dual", "fp"), "single",
    ("单柱", "双柱", "FP"))
BOOL = BoolControlSpec("far_field", "far_field_widget", False)
ML_BOOL = BoolControlSpec("ml_accel", "ml_accel_control", False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (180, 180.0), (50.25, 50.25), (999, 350.0), (-1, 50.0),
        ("legacy", 180.0), (float("nan"), 180.0),
        (float("inf"), 180.0), (float("-inf"), 180.0),
        (True, 180.0), (False, 180.0),
    ],
)
def test_numeric_normalization_is_finite_typed_and_bounded(raw, expected):
    actual = normalize_numeric(raw, NUMERIC)
    assert actual == expected
    assert type(actual) is float
    assert math.isfinite(actual)


def test_enum_and_bool_accept_only_exact_registered_types():
    assert normalize_enum("dual", ENUM) == "dual"
    assert normalize_enum("Dual", ENUM) == "single"
    assert normalize_enum("旧版双柱", ENUM) == "single"
    assert normalize_bool(True, BOOL) is True
    assert normalize_bool(False, BOOL) is False
    assert normalize_bool(1, BOOL) is False
    assert normalize_bool("true", BOOL) is False
    assert normalize_bool(float("nan"), ML_BOOL) is False


def test_migration_normalizes_canonical_and_all_widget_mirrors_idempotently():
    state = {
        "d_val": 999,
        "d_slider": "stale",
        "d_input": float("nan"),
        "structure": "legacy",
        "structure_widget": "双柱",
        "far_field": "true",
        "far_field_widget": True,
    }
    migrate_session_state(state, (NUMERIC,), (ENUM,), (BOOL,))
    first = dict(state)
    migrate_session_state(state, (NUMERIC,), (ENUM,), (BOOL,))

    assert state == first
    assert state["d_val"] == state["d_slider"] == state["d_input"] == 350.0
    assert state["structure"] == "single"
    assert state["structure_widget"] == "单柱"
    assert state["far_field"] is state["far_field_widget"] is False
    assert state[SESSION_MIGRATION_KEY] == {
        "schema": SESSION_MIGRATION_SCHEMA,
        "version": SESSION_MIGRATION_VERSION,
    }


def test_migration_preserves_valid_widget_value_when_canonical_is_absent():
    state = {
        "d_input": 225.5,
        "structure_widget": "双柱",
        "far_field_widget": True,
    }
    migrate_session_state(state, (NUMERIC,), (ENUM,), (BOOL,))
    assert state["d_val"] == state["d_slider"] == state["d_input"] == 225.5
    assert state["structure"] == "dual"
    assert state["structure_widget"] == "双柱"
    assert state["far_field"] is state["far_field_widget"] is True


def test_numeric_callbacks_synchronize_both_directions_and_setter():
    state = {"d_val": 180.0, "d_slider": 240.0, "d_input": 180.0}
    sync_numeric_from_widget(state, NUMERIC, "d_slider")
    assert state["d_val"] == state["d_slider"] == state["d_input"] == 240.0

    state["d_input"] = 275.5
    sync_numeric_from_widget(state, NUMERIC, "d_input")
    assert state["d_val"] == state["d_slider"] == state["d_input"] == 275.5

    set_numeric_value(state, NUMERIC, 999)
    assert state["d_val"] == state["d_slider"] == state["d_input"] == 350.0
    with pytest.raises(ValueError, match="unregistered"):
        sync_numeric_from_widget(state, NUMERIC, "unknown")


def test_enum_and_bool_callbacks_update_canonical_and_widget():
    state = {"structure_widget": "FP", "far_field_widget": True}
    sync_enum_from_widget(state, ENUM)
    sync_bool_from_widget(state, BOOL)
    assert state["structure"] == "fp"
    assert state["structure_widget"] == "FP"
    assert state["far_field"] is state["far_field_widget"] is True

    ml_state = {"ml_accel_control": True}
    sync_bool_from_widget(ml_state, ML_BOOL)
    assert ml_state["ml_accel"] is ml_state["ml_accel_control"] is True


def test_bool_preference_marker_distinguishes_fresh_placeholders_from_legacy():
    fresh = {}
    assert initialize_bool_preference_marker(
        fresh, ML_BOOL, ML_ACCEL_PREFERENCE_INITIALIZED_KEY) is False
    set_bool_value(fresh, ML_BOOL, False)
    assert initialize_bool_preference_marker(
        fresh, ML_BOOL, ML_ACCEL_PREFERENCE_INITIALIZED_KEY) is False

    for raw in (True, False, "legacy", 1, float("nan")):
        legacy = {"ml_accel": raw}
        assert initialize_bool_preference_marker(
            legacy, ML_BOOL, ML_ACCEL_PREFERENCE_INITIALIZED_KEY) is True
        migrate_session_state(legacy, (), (), (ML_BOOL,))
        assert legacy[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True


def test_bool_preference_callback_sets_strict_value_and_initialized_marker():
    state = {
        "ml_accel": False,
        "ml_accel_control": "not-a-bool",
        ML_ACCEL_PREFERENCE_INITIALIZED_KEY: False,
    }
    sync_bool_preference_from_widget(
        state, ML_BOOL, ML_ACCEL_PREFERENCE_INITIALIZED_KEY)
    assert state["ml_accel"] is state["ml_accel_control"] is False
    assert state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True


def test_app_migrates_before_engine_and_sidebar_widgets():
    from pathlib import Path

    source = Path("app.py").read_text(encoding="utf-8")
    migration = source.index("migrate_session_state(")
    engine = source.index("engine = get_engine()")
    sidebar = source.index("with st.sidebar:")
    assert migration < engine < sidebar
    assert ".index(st.session_state" not in source
