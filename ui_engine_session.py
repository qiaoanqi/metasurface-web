"""Session ownership and explicit library identity for mutable UI engines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping
import uuid

import numpy as np


ENGINE_SESSION_CONTRACT_VERSION = "ui-engine-session-v1"
ENGINE_SESSION_KEY = "_ui_engine_instance_v1"
ENGINE_OWNER_KEY = "_ui_engine_owner_v1"
ENGINE_LIBRARY_KEY = "_ui_engine_library_identity_v1"


@dataclass(frozen=True)
class LibraryIdentity:
    """Full identity required before any mutable library/grid is consumed."""

    material: str
    substrate: str
    polarization: str
    angle_deg: float
    far_field_enabled: bool = False
    na: float = 0.1
    theta_obs_deg: float = 0.0

    def __post_init__(self) -> None:
        for name in ("material", "substrate", "polarization"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"empty library identity field: {name}")
        angle_deg = float(self.angle_deg)
        far_field_enabled = bool(self.far_field_enabled)
        if not np.isfinite(angle_deg) or not 0.0 <= angle_deg <= 80.0:
            raise ValueError("angle_deg must be finite and within [0, 80]")

        if far_field_enabled:
            na = float(self.na)
            theta_obs_deg = float(self.theta_obs_deg)
            if not np.isfinite(na) or not 0.05 <= na <= 0.95:
                raise ValueError("na must be finite and within [0.05, 0.95]")
            if not np.isfinite(theta_obs_deg) or not 0.0 <= theta_obs_deg <= 80.0:
                raise ValueError(
                    "theta_obs_deg must be finite and within [0, 80]")
        else:
            na = 0.1
            theta_obs_deg = 0.0

        object.__setattr__(self, "angle_deg", angle_deg)
        object.__setattr__(self, "far_field_enabled", far_field_enabled)
        object.__setattr__(self, "na", na)
        object.__setattr__(self, "theta_obs_deg", theta_obs_deg)

    @property
    def grid_key(self) -> tuple[str, str, str, float]:
        return (
            str(self.material), str(self.substrate), str(self.polarization),
            float(self.angle_deg),
        )


def get_session_engine(
    session_state: MutableMapping[str, Any],
    factory: Callable[[], Any],
) -> Any:
    """Return one mutable engine owned exclusively by this session mapping."""
    owner = session_state.get(ENGINE_OWNER_KEY)
    if not isinstance(owner, str) or not owner:
        owner = uuid.uuid4().hex
        session_state[ENGINE_OWNER_KEY] = owner
    engine = session_state.get(ENGINE_SESSION_KEY)
    valid = bool(
        engine is not None
        and getattr(engine, "_ui_session_contract_version", None)
        == ENGINE_SESSION_CONTRACT_VERSION
        and getattr(engine, "_ui_session_owner", None) == owner
    )
    if not valid:
        engine = factory()
        setattr(engine, "_ui_session_contract_version", ENGINE_SESSION_CONTRACT_VERSION)
        setattr(engine, "_ui_session_owner", owner)
        setattr(engine, "_ui_library_identity", None)
        session_state[ENGINE_SESSION_KEY] = engine
        session_state.pop(ENGINE_LIBRARY_KEY, None)
    return engine


def configure_engine_far_field(
    engine: Any,
    enabled: bool,
    na: float = 0.1,
    theta_obs_deg: float = 0.0,
) -> tuple[bool, float, float]:
    """Apply session-local far-field state; disabled state clears old values."""
    enabled = bool(enabled)
    if enabled:
        na_value = float(na)
        theta_value = float(theta_obs_deg)
        if not np.isfinite(na_value) or not np.isfinite(theta_value):
            raise ValueError("non-finite far-field configuration")
    else:
        na_value = 0.1
        theta_value = 0.0
    engine._enable_far_field = enabled
    engine._na = na_value
    engine._theta_obs_deg = theta_value
    return enabled, na_value, theta_value


def _observed_grid_key(engine: Any) -> tuple[str, str, str, float] | None:
    try:
        return (
            str(engine._last_material), str(engine._last_substrate),
            str(engine._last_polarization), float(engine._last_angle),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _validate_grid_arrays(engine: Any) -> None:
    specifications = {
        "grid_params": 3, "grid_rgb": 3, "grid_lab": 3, "grid_xy": 2,
    }
    lengths = []
    for name, width in specifications.items():
        value = np.asarray(getattr(engine, name, None))
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] == 0:
            raise RuntimeError(f"engine {name} is empty or has invalid shape")
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"engine {name} contains NaN/Inf")
        lengths.append(int(value.shape[0]))
    if len(set(lengths)) != 1:
        raise RuntimeError("engine grid arrays have inconsistent lengths")


def _grid_arrays_valid(engine: Any) -> bool:
    try:
        _validate_grid_arrays(engine)
    except RuntimeError:
        return False
    return True


def bind_engine_library(
    engine: Any,
    identity: LibraryIdentity,
    session_state: MutableMapping[str, Any] | None = None,
) -> LibraryIdentity:
    """Bind and verify the exact library before grid-dependent operations."""
    configure_engine_far_field(
        engine, identity.far_field_enabled, identity.na, identity.theta_obs_deg)
    needs_rebuild = bool(
        _observed_grid_key(engine) != identity.grid_key
        or not _grid_arrays_valid(engine)
    )
    if needs_rebuild:
        engine.rebuild_library(
            identity.material, identity.substrate,
            identity.polarization, identity.angle_deg,
        )
    if _observed_grid_key(engine) != identity.grid_key:
        raise RuntimeError("engine library identity mismatch after rebuild")
    _validate_grid_arrays(engine)
    setattr(engine, "_ui_library_identity", identity)
    if session_state is not None:
        session_state[ENGINE_LIBRARY_KEY] = identity
    return identity


def engine_library_matches(engine: Any, identity: LibraryIdentity) -> bool:
    """Check both the declared marker and observed mutable engine fields."""
    if getattr(engine, "_ui_library_identity", None) != identity:
        return False
    if _observed_grid_key(engine) != identity.grid_key:
        return False
    try:
        _validate_grid_arrays(engine)
    except RuntimeError:
        return False
    return bool(
        getattr(engine, "_enable_far_field", None) == identity.far_field_enabled
        and float(getattr(engine, "_na", np.nan)) == float(identity.na)
        and float(getattr(engine, "_theta_obs_deg", np.nan))
        == float(identity.theta_obs_deg)
    )


def make_local_bound_engine(
    factory: Callable[[], Any],
    identity: LibraryIdentity,
) -> Any:
    """Create an operation-local engine that cannot mutate a session engine."""
    engine = factory()
    bind_engine_library(engine, identity)
    return engine
