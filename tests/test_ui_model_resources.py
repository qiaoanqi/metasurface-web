from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import threading

import pytest

from ui_model_resources import (
    BoundModelResource, ModelResourceDriftError, bound_model_context,
    exception_has_model_resource_drift, get_bound_resource, register_first_resource,
    _reset_resource_registry_for_tests,
)


@dataclass(frozen=True)
class Identity:
    available: bool
    version: str
    reason: str = ""


@pytest.fixture(autouse=True)
def _empty_registry():
    _reset_resource_registry_for_tests()
    yield
    _reset_resource_registry_for_tests()


def _resource(family, version, session, path="models/model.onnx"):
    return BoundModelResource.create(
        family, f"context:{family}", version, (path,), (path,), (session,))


def test_first_binding_keeps_strong_ref_and_rejects_same_path_new_session():
    session_a = object()
    session_b = object()
    bound_a = register_first_resource(
        _resource("generic", "sha256:A", session_a))
    returned = register_first_resource(
        _resource("generic", "sha256:B", session_b))

    assert returned is bound_a
    assert returned.session_refs == (session_a,)
    assert returned.session_refs[0] is session_a
    assert returned.session_ids == (id(session_a),)
    assert get_bound_resource("generic", "context:generic") is bound_a


def test_graph_or_external_data_identity_drift_fails_before_evaluator():
    session = object()
    module = SimpleNamespace(session="global-bomb")
    disk = {"version": "sha256:graph-A+data-A"}
    calls = {"evaluator": 0}
    resource = _resource("generic", disk["version"], session)

    disk["version"] = "sha256:graph-A+data-B"
    with pytest.raises(Exception, match="current model bytes differ"):
        with bound_model_context(
            resource, module, {"session": session},
            lambda: Identity(True, disk["version"]),
            lambda: (module.session,),
            resource.context_key,
        ):
            calls["evaluator"] += 1

    assert calls["evaluator"] == 0
    assert module.session == "global-bomb"


@pytest.mark.parametrize("raise_inside", [False, True])
def test_bound_context_restores_global_on_success_or_error(raise_inside):
    session = object()
    bomb = object()
    module = SimpleNamespace(session=bomb, ready=False)
    resource = _resource("dual", "sha256:A", session)

    with pytest.raises(RuntimeError) if raise_inside else _does_not_raise():
        with bound_model_context(
            resource, module, {"session": session, "ready": True},
            lambda: Identity(True, "sha256:A"),
            lambda: (module.session,),
            resource.context_key,
        ):
            assert module.session is session
            assert module.ready is True
            if raise_inside:
                raise RuntimeError("fixture")

    assert module.session is bomb
    assert module.ready is False


def test_toctou_after_evaluator_restores_global_and_discards_result():
    session = object()
    bomb = object()
    module = SimpleNamespace(session=bomb)
    disk = {"version": "sha256:A"}
    resource = _resource("rcwa", disk["version"], session)

    with pytest.raises(ModelResourceDriftError, match="changed during"):
        with bound_model_context(
            resource, module, {"session": session},
            lambda: Identity(True, disk["version"]),
            lambda: (module.session,),
            resource.context_key,
        ):
            disk["version"] = "sha256:B"

    assert module.session is bomb


def test_session_mutation_after_success_raises_drift_and_restores_ambient():
    session_a = object()
    ambient_b = object()
    mutation_c = object()
    module = SimpleNamespace(session=ambient_b)
    resource = _resource("generic", "sha256:A", session_a)

    with pytest.raises(ModelResourceDriftError, match="runtime sessions changed"):
        with bound_model_context(
            resource, module, {"session": session_a},
            lambda: Identity(True, "sha256:A"),
            lambda: (module.session,), resource.context_key,
        ):
            module.session = mutation_c

    assert module.session is ambient_b


def test_session_mutation_then_business_error_preserves_error_and_restores_ambient():
    session_a = object()
    ambient_b = object()
    mutation_c = object()
    module = SimpleNamespace(session=ambient_b)
    resource = _resource("dual", "sha256:A", session_a)

    with pytest.raises(RuntimeError, match="business") as captured:
        with bound_model_context(
            resource, module, {"session": session_a},
            lambda: Identity(True, "sha256:A"),
            lambda: (module.session,), resource.context_key,
        ):
            module.session = mutation_c
            raise RuntimeError("business")

    assert module.session is ambient_b
    assert any("runtime sessions changed" in note for note in captured.value.__notes__)
    assert exception_has_model_resource_drift(captured.value)


@pytest.mark.parametrize("business_error", [False, True])
def test_post_identity_callback_error_restores_ambient_and_preserves_business_error(
        business_error):
    session = object()
    ambient = object()
    module = SimpleNamespace(session=ambient)
    resource = _resource("generic", "sha256:A", session)
    identity_calls = {"count": 0}

    def identity():
        identity_calls["count"] += 1
        if identity_calls["count"] > 1:
            raise ValueError("post identity probe")
        return Identity(True, "sha256:A")

    if business_error:
        with pytest.raises(RuntimeError, match="business") as captured:
            with bound_model_context(
                resource, module, {"session": session}, identity,
                lambda: (module.session,), resource.context_key,
            ):
                raise RuntimeError("business")
        assert any(
            "identity post-check failed: ValueError" in note
            for note in captured.value.__notes__)
    else:
        with pytest.raises(
                ModelResourceDriftError,
                match="identity post-check failed: ValueError"):
            with bound_model_context(
                resource, module, {"session": session}, identity,
                lambda: (module.session,), resource.context_key,
            ):
                pass

    assert module.session is ambient


def test_two_threads_with_distinct_bindings_are_serialized_without_cross_talk():
    session_a = object()
    session_b = object()
    bomb = object()
    module = SimpleNamespace(session=bomb)
    active = {"count": 0, "max": 0}
    seen = []
    errors = []

    def worker(name, session):
        resource = _resource(name, f"sha256:{name}", session)
        try:
            with bound_model_context(
                resource, module, {"session": session},
                lambda: Identity(True, f"sha256:{name}"),
                lambda: (module.session,),
                resource.context_key,
            ):
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
                seen.append((name, module.session is session))
                threading.Event().wait(0.01)
                active["count"] -= 1
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("A", session_a)),
        threading.Thread(target=worker, args=("B", session_b)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert errors == []
    assert active["max"] == 1
    assert sorted(seen) == [("A", True), ("B", True)]
    assert module.session is bomb


class _does_not_raise:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False
