"""Pure contracts for structure-specific inverse-design UI state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import json
import math
from typing import Any, Iterable, Mapping


_STRUCTURES = {"single", "dual", "fp"}


@dataclass(frozen=True)
class InverseContext:
    structure_type: str
    material: str
    substrate: str
    polarization: str
    angle_deg: float
    target_rgb: tuple[int, int, int]
    target_hex: str
    preview_route_id: str
    preview_model_version: str
    geometry_valid: bool
    fp_mirror_type: str = ""

    def __post_init__(self) -> None:
        if self.structure_type not in _STRUCTURES:
            raise ValueError(f"unsupported inverse structure: {self.structure_type}")
        rgb = tuple(int(value) for value in self.target_rgb)
        if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
            raise ValueError("target_rgb must contain three integers in [0, 255]")
        normalized_hex = str(self.target_hex).strip().upper()
        if len(normalized_hex) != 7 or not normalized_hex.startswith("#"):
            raise ValueError("target_hex must use #RRGGBB")
        try:
            hex_rgb = tuple(int(normalized_hex[index:index + 2], 16) for index in (1, 3, 5))
        except ValueError as exc:
            raise ValueError("target_hex must use #RRGGBB") from exc
        if hex_rgb != rgb:
            raise ValueError("target_hex and target_rgb disagree")
        object.__setattr__(self, "target_rgb", rgb)
        object.__setattr__(self, "target_hex", normalized_hex)
        object.__setattr__(self, "angle_deg", float(self.angle_deg))
        object.__setattr__(self, "geometry_valid", bool(self.geometry_valid))
        object.__setattr__(self, "fp_mirror_type", str(self.fp_mirror_type))


@dataclass(frozen=True)
class InverseRun:
    context: InverseContext
    method_id: str
    method_label: str
    candidates: tuple[Any, ...]
    contract_version: str = "inverse-ui-v2"

    @classmethod
    def create(
        cls,
        context: InverseContext,
        method_id: str,
        method_label: str,
        candidates: Iterable[Any],
    ) -> "InverseRun":
        return cls(context, str(method_id), str(method_label), tuple(candidates))


@dataclass(frozen=True)
class InverseMethodState:
    method_id: str
    label: str
    summary: str
    available: bool
    reason: str
    scope: str = "structure"


@dataclass(frozen=True)
class InverseExports:
    csv_text: str
    json_text: str
    payload: Mapping[str, Any]


_CANDIDATE_REQUIRED_FIELDS = {
    "rank", "structure_type", "method_id", "method_label",
    "context_fingerprint", "candidate_context", "route_id", "route_label",
    "model_version", "boundary", "parameters", "predicted_rgb",
    "predicted_rgb255", "predicted_hex", "delta_e2000",
}


def _finite_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def build_inverse_candidate(
    context: InverseContext,
    *,
    method_id: str,
    method_label: str,
    rank: int,
    structure_type: str,
    candidate_context: Mapping[str, Any],
    route_id: str,
    route_label: str,
    model_version: str,
    boundary: str,
    parameters: Mapping[str, Any],
    predicted_rgb: Iterable[Any],
    delta_e2000: Any,
    delta_e76: Any | None = None,
    scheme: str = "",
) -> dict[str, Any]:
    """Build one self-contained inverse candidate without guessing missing fields."""
    structure = str(structure_type)
    if structure not in _STRUCTURES:
        raise ValueError(f"unsupported candidate structure: {structure}")
    method_id = str(method_id).strip()
    method_label = str(method_label).strip()
    route_id = str(route_id).strip()
    route_label = str(route_label).strip()
    model_version = str(model_version).strip()
    boundary = str(boundary).strip()
    if not all((method_id, method_label, route_id, route_label, model_version, boundary)):
        raise ValueError("candidate method, route, model, and boundary are required")

    try:
        rank_value = int(rank)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate rank must be a positive integer") from exc
    if rank_value < 1 or rank_value != rank:
        raise ValueError("candidate rank must be a positive integer")

    context_payload = dict(candidate_context)
    required_context = {
        "structure_type", "material", "substrate", "polarization", "angle_deg",
    }
    if not required_context.issubset(context_payload):
        raise ValueError("candidate_context is incomplete")
    if str(context_payload["structure_type"]) != structure:
        raise ValueError("candidate_context structure does not match candidate")
    context_payload["angle_deg"] = _finite_float(
        context_payload["angle_deg"], "candidate_context.angle_deg")
    for key in ("material", "substrate", "polarization"):
        context_payload[key] = str(context_payload[key]).strip()
        if not context_payload[key]:
            raise ValueError(f"candidate_context.{key} is required")

    parameter_payload = {
        str(key): _finite_float(value, f"parameters.{key}")
        for key, value in dict(parameters).items()
    }
    required_parameters = {
        "single": {"d", "h", "p"},
        "dual": {"d1", "h1", "d2", "h2", "p"},
        "fp": {"t", "center_wavelength"},
    }[structure]
    if not required_parameters.issubset(parameter_payload):
        raise ValueError(f"parameters are incomplete for {structure}")

    rgb = tuple(_finite_float(value, "predicted_rgb") for value in predicted_rgb)
    if len(rgb) != 3 or any(value < 0.0 or value > 1.0 for value in rgb):
        raise ValueError("predicted_rgb must contain three values in [0, 1]")
    # Match preview and PNG quantization, including half-to-even rounding.
    rgb255 = tuple(max(0, min(255, round(value * 255))) for value in rgb)
    record: dict[str, Any] = {
        "rank": rank_value,
        "structure_type": structure,
        "method_id": method_id,
        "method_label": method_label,
        "context_fingerprint": inverse_context_fingerprint(context),
        "candidate_context": context_payload,
        "route_id": route_id,
        "route_label": route_label,
        "model_version": model_version,
        "boundary": boundary,
        "parameters": parameter_payload,
        "predicted_rgb": list(rgb),
        "predicted_rgb255": list(rgb255),
        "predicted_hex": f"#{rgb255[0]:02X}{rgb255[1]:02X}{rgb255[2]:02X}",
        "delta_e2000": _finite_float(delta_e2000, "delta_e2000"),
    }
    if record["delta_e2000"] < 0:
        raise ValueError("delta_e2000 must be non-negative")
    if delta_e76 is not None:
        record["delta_e76"] = _finite_float(delta_e76, "delta_e76")
        if record["delta_e76"] < 0:
            raise ValueError("delta_e76 must be non-negative")
    if scheme:
        record["scheme"] = str(scheme)
    return record


def fp_search_cache_key(
    context: InverseContext,
    *,
    mirror_type: str,
    algorithm_version: str = "fp-dbr-grid-v1",
) -> str:
    """Bind cached FP candidates to the complete versioned search identity."""
    if context.structure_type != "fp":
        raise ValueError("FP cache identity requires an FP context")
    mirror = str(mirror_type).strip()
    if not mirror or mirror != context.fp_mirror_type:
        raise ValueError("FP mirror identity is missing or inconsistent")
    identity = {
        "context": asdict(context),
        "algorithm_version": str(algorithm_version),
        "mirror_type": mirror,
        "coarse": {"wavelength_nm": [380, 780, 20], "thickness_nm": [50, 600, 20]},
        "fine": {"radius_nm": 18, "step_nm": 4},
        "tmm": {"n_pairs": 3, "layers": 5},
    }
    encoded = json.dumps(
        identity, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inverse_context_fingerprint(context: InverseContext) -> str:
    payload = json.dumps(
        asdict(context), ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inverse_run_matches(run: InverseRun | None, context: InverseContext) -> bool:
    return bool(
        isinstance(run, InverseRun)
        and run.candidates
        and inverse_context_fingerprint(run.context) == inverse_context_fingerprint(context)
    )


def invalidate_inverse_run(
    run: InverseRun | None,
    context: InverseContext,
) -> InverseRun | None:
    return run if inverse_run_matches(run, context) else None


def serialize_inverse_run(
    run: InverseRun,
    current_context: InverseContext,
) -> InverseExports:
    """Serialize only a run valid for the supplied current context."""
    if not inverse_run_matches(run, current_context):
        raise ValueError("inverse run is not valid for the current context")
    candidates: list[dict[str, Any]] = []
    expected_fingerprint = inverse_context_fingerprint(run.context)
    seen_ranks: set[int] = set()
    for candidate in run.candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("inverse run candidates must be normalized mappings")
        record = dict(candidate)
        missing = _CANDIDATE_REQUIRED_FIELDS.difference(record)
        if missing:
            raise ValueError(
                f"inverse candidate missing required fields: {sorted(missing)}")
        if record["method_id"] != run.method_id or record["method_label"] != run.method_label:
            raise ValueError("inverse candidate method does not match its run")
        if record["context_fingerprint"] != expected_fingerprint:
            raise ValueError("inverse candidate context fingerprint does not match its run")
        if run.method_id != "compare" and record["structure_type"] != run.context.structure_type:
            raise ValueError("inverse candidate structure does not match its run")
        rank = int(record["rank"])
        if rank in seen_ranks:
            raise ValueError("inverse candidate ranks must be unique")
        seen_ranks.add(rank)
        # Rebuild the record through the strict constructor to validate nested values.
        validated = build_inverse_candidate(
            run.context,
            method_id=record["method_id"], method_label=record["method_label"],
            rank=rank, structure_type=record["structure_type"],
            candidate_context=record["candidate_context"],
            route_id=record["route_id"], route_label=record["route_label"],
            model_version=record["model_version"], boundary=record["boundary"],
            parameters=record["parameters"], predicted_rgb=record["predicted_rgb"],
            delta_e2000=record["delta_e2000"],
            delta_e76=record.get("delta_e76"), scheme=record.get("scheme", ""),
        )
        if validated != record:
            raise ValueError("inverse candidate contains inconsistent derived fields")
        candidates.append(validated)
    context_payload = asdict(run.context)
    payload = {
        "schema_version": 1,
        "contract_version": run.contract_version,
        "context_fingerprint": expected_fingerprint,
        "context": context_payload,
        "method": {"id": run.method_id, "label": run.method_label},
        "candidates": candidates,
    }
    candidate_fields: list[str] = []
    parameter_fields: list[str] = []
    for candidate in candidates:
        for key in candidate:
            if key != "parameters" and key not in candidate_fields:
                candidate_fields.append(key)
        for key in candidate["parameters"]:
            parameter_key = f"parameter_{key}"
            if parameter_key not in parameter_fields:
                parameter_fields.append(parameter_key)
    context_fields = [f"context_{key}" for key in context_payload]
    fieldnames = [
        "contract_version", "context_fingerprint", "method_id", "method_label",
        *context_fields, *candidate_fields, *parameter_fields,
    ]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()

    def csv_value(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return value

    for candidate in candidates:
        row = {
            "contract_version": run.contract_version,
            "context_fingerprint": payload["context_fingerprint"],
            "method_id": run.method_id,
            "method_label": run.method_label,
            **{
                f"context_{key}": csv_value(value)
                for key, value in context_payload.items()
            },
            **{
                key: csv_value(value)
                for key, value in candidate.items() if key != "parameters"
            },
            **{
                f"parameter_{key}": csv_value(value)
                for key, value in candidate["parameters"].items()
            },
        }
        writer.writerow(row)
    return InverseExports(
        csv_text=csv_buffer.getvalue(),
        json_text=json.dumps(payload, ensure_ascii=False, indent=2),
        payload=payload,
    )


def inverse_method_registry(
    context: InverseContext,
    *,
    supported_material: bool,
    supported_substrate: bool,
    rcwa_ready: bool,
    smart_weights_ready: bool,
    primary_torch_ready: bool,
    dual_ready: bool,
    dual_domain_verified: bool,
    compare_enabled: bool,
    rl_ready: bool = False,
    fp_mirror_type: str = "",
) -> dict[str, InverseMethodState]:
    """Return methods belonging to the selected structure, failing closed."""
    common_ok = bool(
        context.geometry_valid and supported_material and supported_substrate)
    normal_incidence = abs(float(context.angle_deg)) < 1e-9
    is_te = str(context.polarization).startswith("TE")

    if context.structure_type == "single":
        smart_ok = bool(
            common_ok and rcwa_ready and smart_weights_ready
            and normal_incidence and is_te)
        single_ok = bool(common_ok and primary_torch_ready and normal_incidence and is_te)
        states = {
            "smart": InverseMethodState(
                "smart", "智能网格", "两阶段 RCWA 训练代理搜索", smart_ok,
                "可用：本地 RCWA 代理与批量搜索权重已就绪。" if smart_ok else
                "不可用：当前几何无效。" if not context.geometry_valid else
                "不可用：当前材料或衬底不在该代理模型的注册范围。" if not (supported_material and supported_substrate) else
                "不可用：缺少 smart 代理对 TM 或非 0° 入射的版本化训练域证据。" if not (is_te and normal_incidence) else
                "不可用：未加载匹配的 RCWA 代理模型或批量推理权重。",
            ),
            "single": InverseMethodState(
                "single", "单柱梯度", "本地 PyTorch 代理梯度优化", single_ok,
                "可用：使用本地单柱 PyTorch 前向模型。" if single_ok else
                "不可用：当前几何无效。" if not context.geometry_valid else
                "不可用：当前材料或衬底不在该代理模型的注册范围。" if not (supported_material and supported_substrate) else
                "不可用：单柱梯度当前仅注册 TE、0° 入射。" if not (is_te and normal_incidence) else
                "不可用：缺少可加载的本地单柱 PyTorch 模型。",
            ),
        }
        if compare_enabled:
            states["compare"] = InverseMethodState(
                "compare", "跨结构方案对比", "TiO2 / a-Si 解析候选与 FP-TMM 对比",
                bool(context.geometry_valid),
                "可用：这是跨结构比较，不作为当前结构的推荐主方法。" if context.geometry_valid
                else "不可用：当前几何无效。",
                scope="cross_structure",
            )
        # RL is an optional route kept out of the legacy registry unless its
        # local q-table and runtime model binding have both been verified.
        # The default keeps the pure contract tests backwards-compatible while
        # the application can expose the real route when it is runnable.
        if rl_ready:
            states["rl"] = InverseMethodState(
                "rl", "RL Q-learning", "本地 Q-table 离散探索",
                bool(smart_ok and context.geometry_valid),
                "可用：本地 q-table 与 RCWA 代理 session 已绑定；结果建议再做前向复核。"
                if smart_ok and context.geometry_valid else
                "不可用：RL 需要同一材料/衬底的 TE、0° RCWA 代理 session。",
                scope="local_qtable",
            )
        return states

    if context.structure_type == "dual":
        dual_ok = bool(
            context.geometry_valid and dual_ready and dual_domain_verified
            and context.substrate == "SiO2 (fused silica)"
            and is_te and normal_incidence)
        return {
            "dual": InverseMethodState(
                "dual", "双柱梯度", "双柱五参数联合优化", dual_ok,
                "可用：双柱 ONNX 模型已加载；当前注册 TE、SiO2 衬底。" if dual_ok else
                "不可用：当前双柱几何无效。" if not context.geometry_valid else
                "不可用：双柱梯度当前仅注册 TE、0° 入射。" if not (is_te and normal_incidence) else
                "不可用：缺少双柱模型的版本化训练域 manifest，不能证明材料与角度覆盖。" if not dual_domain_verified else
                "不可用：双柱 ONNX 模型未加载，或当前衬底不是 SiO2。",
            )
        }

    is_dbr = str(fp_mirror_type).startswith("介质")
    fp_ok = bool(context.geometry_valid and is_dbr)
    return {
        "fp": InverseMethodState(
            "fp", "FP 腔搜索", "DBR 中心波长与腔长 T 联合扫描", fp_ok,
            "可用：使用 FP cavity TMM 搜索 DBR 中心波长与腔长。" if fp_ok else
            "不可用：当前 FP 几何无效。" if not context.geometry_valid else
            "不可用：Ag 反射镜尚无专属逆设计算法；不会误用 DBR 搜索。",
        )
    }


def candidate_parameter_updates(
    context: InverseContext,
    candidate_structure: str,
    parameters: Mapping[str, Any],
) -> dict[str, float]:
    """Translate candidate parameters to structure-specific session keys."""
    structure = str(candidate_structure)
    if structure != context.structure_type:
        raise ValueError(
            f"candidate structure {structure} does not match {context.structure_type}")
    required = {
        "single": (("d", "d_val"), ("h", "h_val"), ("p", "p_val")),
        "dual": (
            ("d1", "d1_val"), ("h1", "h1_val"),
            ("d2", "d2_val"), ("h2", "h2_val"), ("p", "p_val"),
        ),
        "fp": (("t", "fp_t_val"),),
    }[structure]
    updates: dict[str, float] = {}
    for source_key, session_key in required:
        value = float(parameters[source_key])
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"non-finite candidate parameter: {source_key}")
        updates[session_key] = value
    if structure == "fp" and parameters.get("center_wavelength") is not None:
        updates["fp_target_wl"] = float(parameters["center_wavelength"])
    return updates
