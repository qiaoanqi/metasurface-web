"""Versioned Streamlit session normalization and paired-widget synchronization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Any, Mapping, MutableMapping, Sequence


SESSION_MIGRATION_KEY = "_ui_session_migration_v1"
SESSION_MIGRATION_SCHEMA = "ui-session-controls"
SESSION_MIGRATION_VERSION = 1
ML_ACCEL_PREFERENCE_INITIALIZED_KEY = "_ml_accel_preference_initialized_v1"
_MISSING = object()


@dataclass(frozen=True)
class NumericControlSpec:
    canonical_key: str
    widget_keys: tuple[str, ...]
    minimum: float
    maximum: float
    default: float
    value_type: type = float

    def __post_init__(self) -> None:
        if not self.widget_keys:
            raise ValueError("numeric control requires at least one widget key")
        if self.value_type not in (float, int):
            raise ValueError("numeric value_type must be float or int")
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError("numeric default is outside the widget range")


@dataclass(frozen=True)
class EnumControlSpec:
    canonical_key: str
    widget_key: str
    options: tuple[str, ...]
    default: str
    widget_values: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.options or self.default not in self.options:
            raise ValueError("enum default must be one of the canonical options")
        if self.widget_values is not None and len(self.widget_values) != len(self.options):
            raise ValueError("enum widget values must match canonical options")

    @property
    def displayed_values(self) -> tuple[str, ...]:
        return self.widget_values or self.options

    def to_widget(self, value: str) -> str:
        return self.displayed_values[self.options.index(value)]

    def from_widget(self, value: Any) -> str:
        if value not in self.displayed_values:
            return self.default
        return self.options[self.displayed_values.index(value)]


@dataclass(frozen=True)
class BoolControlSpec:
    canonical_key: str
    widget_key: str
    default: bool


def normalize_numeric(value: Any, spec: NumericControlSpec) -> float | int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        value = spec.default
    else:
        value = float(value)
        if not math.isfinite(value):
            value = spec.default
    value = min(float(spec.maximum), max(float(spec.minimum), float(value)))
    return int(value) if spec.value_type is int else float(value)


def normalize_enum(value: Any, spec: EnumControlSpec) -> str:
    return value if isinstance(value, str) and value in spec.options else spec.default


def normalize_bool(value: Any, spec: BoolControlSpec) -> bool:
    return value if type(value) is bool else spec.default


def set_numeric_value(
    state: MutableMapping[str, Any], spec: NumericControlSpec, value: Any,
) -> float | int:
    normalized = normalize_numeric(value, spec)
    state[spec.canonical_key] = normalized
    for key in spec.widget_keys:
        state[key] = normalized
    return normalized


def sync_numeric_from_widget(
    state: MutableMapping[str, Any], spec: NumericControlSpec, source_key: str,
) -> None:
    if source_key not in spec.widget_keys:
        raise ValueError(f"unregistered numeric widget key: {source_key}")
    set_numeric_value(state, spec, state.get(source_key, spec.default))


def set_enum_value(
    state: MutableMapping[str, Any], spec: EnumControlSpec, value: Any,
) -> str:
    normalized = normalize_enum(value, spec)
    state[spec.canonical_key] = normalized
    state[spec.widget_key] = spec.to_widget(normalized)
    return normalized


def sync_enum_from_widget(
    state: MutableMapping[str, Any], spec: EnumControlSpec,
) -> None:
    set_enum_value(state, spec, spec.from_widget(state.get(spec.widget_key)))


def set_bool_value(
    state: MutableMapping[str, Any], spec: BoolControlSpec, value: Any,
) -> bool:
    normalized = normalize_bool(value, spec)
    state[spec.canonical_key] = normalized
    state[spec.widget_key] = normalized
    return normalized


def sync_bool_from_widget(
    state: MutableMapping[str, Any], spec: BoolControlSpec,
) -> None:
    set_bool_value(state, spec, state.get(spec.widget_key, spec.default))


def initialize_bool_preference_marker(
    state: MutableMapping[str, Any], spec: BoolControlSpec, marker_key: str,
) -> bool:
    """Preserve a strict marker or classify pre-migration preference keys as legacy."""
    if marker_key in state and type(state.get(marker_key)) is bool:
        initialized = state[marker_key]
    else:
        initialized = spec.canonical_key in state or spec.widget_key in state
    state[marker_key] = bool(initialized)
    return bool(initialized)


def sync_bool_preference_from_widget(
    state: MutableMapping[str, Any], spec: BoolControlSpec, marker_key: str,
) -> None:
    sync_bool_from_widget(state, spec)
    state[marker_key] = True


def _initial_value(
    state: Mapping[str, Any], canonical_key: str, widget_keys: Sequence[str], default: Any,
) -> Any:
    value = state.get(canonical_key, _MISSING)
    if value is not _MISSING:
        return value
    for key in widget_keys:
        value = state.get(key, _MISSING)
        if value is not _MISSING:
            return value
    return default


def migrate_session_state(
    state: MutableMapping[str, Any],
    numeric_specs: Sequence[NumericControlSpec],
    enum_specs: Sequence[EnumControlSpec],
    bool_specs: Sequence[BoolControlSpec],
) -> None:
    """Normalize canonical state and every registered widget mirror, idempotently."""
    for spec in enum_specs:
        raw = _initial_value(
            state, spec.canonical_key, (spec.widget_key,), spec.default)
        if spec.canonical_key not in state and raw in spec.displayed_values:
            raw = spec.from_widget(raw)
        set_enum_value(state, spec, raw)

    for spec in numeric_specs:
        raw = _initial_value(
            state, spec.canonical_key, spec.widget_keys, spec.default)
        set_numeric_value(state, spec, raw)

    for spec in bool_specs:
        raw = _initial_value(
            state, spec.canonical_key, (spec.widget_key,), spec.default)
        set_bool_value(state, spec, raw)

    state[SESSION_MIGRATION_KEY] = {
        "schema": SESSION_MIGRATION_SCHEMA,
        "version": SESSION_MIGRATION_VERSION,
    }
