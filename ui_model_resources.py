"""Process-stable runtime model bindings for Streamlit UI routes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any, Callable, Mapping, Sequence


class ModelResourceUnavailable(RuntimeError):
    """Raised when a runtime resource cannot be used without identity drift."""


class ModelResourceDriftError(ModelResourceUnavailable):
    """Raised after a bound operation when its disk identity changed."""


_DRIFT_NOTE_MARKER = "model resource changed during bound operation:"


def exception_has_model_resource_drift(exc: BaseException) -> bool:
    """Recognize drift even when a business exception remains primary."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelResourceUnavailable):
            return True
        if any(
                _DRIFT_NOTE_MARKER in str(note)
                for note in getattr(current, "__notes__", ())):
            return True
        current = current.__cause__ or current.__context__
    return False


@dataclass(frozen=True)
class BoundModelResource:
    family: str
    context_key: str
    loaded_identity: str
    registered_paths: tuple[str, ...]
    session_paths: tuple[str, ...]
    session_refs: tuple[object, ...]
    session_ids: tuple[int, ...]

    @classmethod
    def create(
        cls, family: str, context_key: str, loaded_identity: str,
        registered_paths: Sequence[str], session_paths: Sequence[str],
        sessions: Sequence[object],
    ) -> "BoundModelResource":
        refs = tuple(sessions)
        if not family or not context_key or not str(loaded_identity).startswith("sha256:"):
            raise ValueError("bound model identity must be an available SHA-256 identity")
        registered = tuple(sorted(dict.fromkeys(map(str, registered_paths))))
        raw_paths = tuple(map(str, session_paths))
        if not registered or not raw_paths or not refs:
            raise ValueError("bound model resource requires paths and strong session refs")
        if len(raw_paths) != len(refs):
            raise ValueError("session paths and refs must have equal lengths")
        pairs = sorted(zip(raw_paths, refs), key=lambda item: item[0])
        paths = tuple(path for path, _session in pairs)
        refs = tuple(session for _path, session in pairs)
        if len(set(paths)) != len(paths):
            raise ValueError("bound model session paths must be unique")
        return cls(
            str(family), str(context_key), str(loaded_identity), registered, paths, refs,
            tuple(id(session) for session in refs),
        )


_RESOURCE_LOCK = threading.RLock()
_RESOURCE_REGISTRY: dict[tuple[str, str], BoundModelResource] = {}


def resource_lock() -> threading.RLock:
    return _RESOURCE_LOCK


def get_bound_resource(family: str, context_key: str) -> BoundModelResource | None:
    with _RESOURCE_LOCK:
        return _RESOURCE_REGISTRY.get((str(family), str(context_key)))


def register_first_resource(resource: BoundModelResource) -> BoundModelResource:
    """Register exactly one resource per route slot; never hot-reload it."""
    with _RESOURCE_LOCK:
        key = (resource.family, resource.context_key)
        existing = _RESOURCE_REGISTRY.get(key)
        if existing is None:
            _RESOURCE_REGISTRY[key] = resource
            return resource
        return existing


def validate_bound_resource(
    resource: BoundModelResource | None,
    *, current_available: bool,
    current_identity: str,
    current_reason: str = "",
    expected_context_key: str | None = None,
) -> str:
    if resource is None:
        return "runtime model is not bound"
    if not current_available:
        return str(current_reason or "current model bytes are unavailable")
    if str(current_identity) != resource.loaded_identity:
        return "current model bytes differ from the load-time identity"
    if tuple(id(session) for session in resource.session_refs) != resource.session_ids:
        return "bound session object identity changed"
    if expected_context_key is not None and resource.context_key != str(expected_context_key):
        return "bound model context differs from the requested route context"
    return ""


@contextmanager
def bound_model_context(
    resource: BoundModelResource,
    module: object,
    install_globals: Mapping[str, Any],
    current_identity: Callable[[], Any],
    current_sessions: Callable[[], Sequence[object]],
    expected_context_key: str,
):
    """Serialize one bound operation, install exact refs, and always restore globals."""
    with _RESOURCE_LOCK:
        before = current_identity()
        issue = validate_bound_resource(
            resource, current_available=bool(getattr(before, "available", False)),
            current_identity=str(getattr(before, "version", "")),
            current_reason=str(getattr(before, "reason", "")),
            expected_context_key=expected_context_key,
        )
        if issue:
            raise ModelResourceUnavailable(issue)
        originals = {
            name: getattr(module, name) for name in install_globals
        }
        operation_error: BaseException | None = None
        post_session_issue = ""
        try:
            for name, value in install_globals.items():
                setattr(module, name, value)
            active = tuple(current_sessions())
            if (
                len(active) != len(resource.session_refs)
                or any(actual is not bound for actual, bound in zip(
                    active, resource.session_refs))
            ):
                raise ModelResourceUnavailable(
                    "installed runtime sessions differ from the bound resource")
            yield resource
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            try:
                active_after = tuple(current_sessions())
                if (
                    len(active_after) != len(resource.session_refs)
                    or any(actual is not bound for actual, bound in zip(
                        active_after, resource.session_refs))
                ):
                    post_session_issue = (
                        "runtime sessions changed during bound operation")
            except BaseException as exc:
                post_session_issue = (
                    f"runtime session post-check failed: {type(exc).__name__}")
            for name, value in originals.items():
                setattr(module, name, value)
            try:
                after = current_identity()
                post_issue = validate_bound_resource(
                    resource, current_available=bool(getattr(after, "available", False)),
                    current_identity=str(getattr(after, "version", "")),
                    current_reason=str(getattr(after, "reason", "")),
                    expected_context_key=expected_context_key,
                )
            except BaseException as exc:
                post_issue = (
                    f"runtime identity post-check failed: {type(exc).__name__}")
            drift_reason = post_session_issue or post_issue
            if drift_reason:
                drift = ModelResourceDriftError(
                    f"{_DRIFT_NOTE_MARKER} {drift_reason}")
                if operation_error is None:
                    raise drift
                try:
                    operation_error.add_note(str(drift))
                except (AttributeError, TypeError):
                    pass


def _reset_resource_registry_for_tests() -> None:
    """Test-only reset; application code must never clear process bindings."""
    with _RESOURCE_LOCK:
        _RESOURCE_REGISTRY.clear()
