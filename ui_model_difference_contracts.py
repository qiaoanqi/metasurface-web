"""Pure contracts for the Fano-to-generic-ONNX difference analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from color_utils import WL, delta_e2000, rgb_to_lab, spectrum_to_srgb


@dataclass(frozen=True)
class PairCoverage:
    material: str
    substrate: str
    train_count: int
    validation_count: int

    @property
    def fully_trained(self) -> bool:
        return self.train_count == 13_000


@dataclass(frozen=True)
class ConversionEvidence:
    protocol_path: str
    result_path: str
    protocol_sha256: str
    result_sha256: str
    audit_script_sha256: str
    sample_count: int
    input_sha256: str
    max_abs_diff: float
    tolerance: float


@dataclass(frozen=True)
class GenericOnnxRouteContract:
    route_id: str
    model_version: str
    model_relative_path: str
    external_data_relative_path: str
    source_pt_relative_path: str
    model_sha256: str
    external_data_sha256: str
    source_pt_sha256: str
    training_commit: str
    conversion_commit: str
    conversion_protocol_relative_path: str
    conversion_result_relative_path: str
    conversion_protocol_sha256: str
    conversion_result_sha256: str
    materials: tuple[str, ...]
    substrates: tuple[str, ...]
    polarizations: tuple[str, ...]
    pair_coverages: tuple[PairCoverage, ...]
    angle_range_deg: tuple[float, float]
    d_range_nm: tuple[float, float]
    h_range_nm: tuple[float, float]
    p_range_nm: tuple[float, float]

    @property
    def boundary_text(self) -> str:
        return (
            "训练证据域按材料/衬底 pair 显式注册，只有 train=13,000 的 pair 放行；"
            "目标生成函数未使用偏振输入，本分析只允许 TE，不能比较 TM 物理差异；入射角 "
            f"{self.angle_range_deg[0]:.0f}-{self.angle_range_deg[1]:.0f}°；"
            f"D={self.d_range_nm[0]:.0f}-{self.d_range_nm[1]:.0f} nm，"
            f"H={self.h_range_nm[0]:.0f}-{self.h_range_nm[1]:.0f} nm，"
            f"P={self.p_range_nm[0]:.0f}-{self.p_range_nm[1]:.0f} nm 且 P≥1.2D。"
        )

    def evidence_text(self, evidence: ConversionEvidence) -> str:
        return (
            f"训练提交 {self.training_commit}；转换提交 {self.conversion_commit}；"
            f"source PT SHA-256={self.source_pt_sha256}；"
            f"转换协议 {evidence.protocol_path}（canonical JSON SHA-256={evidence.protocol_sha256}）；"
            f"版本化结果 {evidence.result_path}（canonical JSON SHA-256={evidence.result_sha256}）；"
            f"转换一致性重算 N={evidence.sample_count}，input SHA-256={evidence.input_sha256}，"
            f"PT-vs-ONNX max|Δ|={evidence.max_abs_diff:.17g}，"
            f"容差={evidence.tolerance:.1e}。"
        )

    def pair_coverage(self, material: str, substrate: str) -> PairCoverage | None:
        return next((
            item for item in self.pair_coverages
            if item.material == material and item.substrate == substrate
        ), None)

    def pair_evidence_text(self, material: str, substrate: str) -> str:
        coverage = self.pair_coverage(material, substrate)
        if coverage is None:
            return f"pair {material} / {substrate} 未注册训练计数"
        decision = "放行" if coverage.fully_trained else "禁用，不输出指标"
        return (
            f"当前 pair：{material} / {substrate}；train={coverage.train_count:,}，"
            f"validation={coverage.validation_count:,}；保守规则：{decision}。"
        )

    def context_issue(
        self, material: str, substrate: str, polarization: str, angle_deg: float,
    ) -> str:
        coverage = self.pair_coverage(material, substrate)
        if coverage is None:
            return f"材料/衬底 pair {material} / {substrate} 未注册训练证据"
        if not coverage.fully_trained:
            return (
                f"材料/衬底 pair {material} / {substrate} 训练覆盖不足："
                f"train={coverage.train_count:,}，validation={coverage.validation_count:,}；"
                "按保守规则禁用")
        pol = "TE" if str(polarization).startswith("TE") else (
            "TM" if str(polarization).startswith("TM") else str(polarization))
        if pol not in self.polarizations:
            return (
                f"偏振 {polarization} 不可用：训练 target single_spectrum 未使用偏振输入，"
                "不能比较 TM 物理差异")
        angle = float(angle_deg)
        if not math.isfinite(angle) or not (
            self.angle_range_deg[0] <= angle <= self.angle_range_deg[1]
        ):
            return f"入射角 {angle_deg}° 不在冻结的 generic ONNX 训练证据域内"
        return ""

    def geometry_issue(self, geometries: object) -> str:
        try:
            values = np.asarray(geometries, dtype=float)
        except (TypeError, ValueError):
            return "几何数组无法转换为有限浮点数"
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 1:
            return "几何数组必须为非空 (N, 3) 的 D/H/P"
        if not np.all(np.isfinite(values)):
            return "几何数组包含 NaN/Inf"
        d_nm, h_nm, p_nm = values.T
        if np.any((d_nm < self.d_range_nm[0]) | (d_nm > self.d_range_nm[1])):
            return "D 超出冻结训练证据域"
        if np.any((h_nm < self.h_range_nm[0]) | (h_nm > self.h_range_nm[1])):
            return "H 超出冻结训练证据域"
        if np.any((p_nm < self.p_range_nm[0]) | (p_nm > self.p_range_nm[1])):
            return "P 超出冻结训练证据域"
        if np.any(p_nm < 1.2 * d_nm):
            return "存在 P<1.2D 的样本，超出冻结训练证据域"
        return ""


GENERIC_ONNX_ROUTE = GenericOnnxRouteContract(
    route_id="generic-fano-resmlp-v8-sub-onnx-only",
    model_version="forward_mlp_v8_sub / 7-input / 81-output",
    model_relative_path="models/forward_mlp_v8_sub.onnx",
    external_data_relative_path="models/forward_mlp_v8_sub.onnx.data",
    source_pt_relative_path="models/forward_mlp_v8_sub.pt",
    model_sha256="ddf41cfe8cbdbab2dbb669ab6c39c10fffe7b8fab3126cdc13bee6a6a6a9aede",
    external_data_sha256="f9572fe9d39d0db6fabda60a0e8bb5eef808972b2786bbac08610ca1ee0e39a4",
    source_pt_sha256="01fb06258dd941a01f8adac71101b998376821ce8600d1a0745d21e7dfa5fd47",
    training_commit="2e8c170",
    conversion_commit="64b94de",
    conversion_protocol_relative_path="protocols/forward_mlp_v8_sub_conversion_v1.json",
    conversion_result_relative_path="models/evidence/forward_mlp_v8_sub_conversion_v1.json",
    conversion_protocol_sha256="c7c4e5ff47731dfaa3020a52a3159a9bc6467b3c6b2bf9dd149d7acbae01ce40",
    conversion_result_sha256="3ff670fb2a42a524618fdd4c973033dcf418456a3de0b5d1db8758df33bd2179",
    materials=(
        "TiO2 (anatase)", "a-Si (amorphous)",
        "Si3N4 (nitride)", "Al2O3 (sapphire)",
    ),
    substrates=(
        "SiO2 (fused silica)", "Si3N4 (nitride)", "Al2O3 (sapphire)",
    ),
    polarizations=("TE",),
    # train_v8_substrate.py writes 13,000 rows per pair in this exact order,
    # then takes the first 85% as train without shuffling the full array.
    pair_coverages=tuple(
        PairCoverage(material, substrate, train_count, validation_count)
        for material, substrate, train_count, validation_count in (
            ("TiO2 (anatase)", "SiO2 (fused silica)", 13_000, 0),
            ("TiO2 (anatase)", "Si3N4 (nitride)", 13_000, 0),
            ("TiO2 (anatase)", "Al2O3 (sapphire)", 13_000, 0),
            ("a-Si (amorphous)", "SiO2 (fused silica)", 13_000, 0),
            ("a-Si (amorphous)", "Si3N4 (nitride)", 13_000, 0),
            ("a-Si (amorphous)", "Al2O3 (sapphire)", 13_000, 0),
            ("Si3N4 (nitride)", "SiO2 (fused silica)", 13_000, 0),
            ("Si3N4 (nitride)", "Si3N4 (nitride)", 13_000, 0),
            ("Si3N4 (nitride)", "Al2O3 (sapphire)", 13_000, 0),
            ("Al2O3 (sapphire)", "SiO2 (fused silica)", 13_000, 0),
            ("Al2O3 (sapphire)", "Si3N4 (nitride)", 2_600, 10_400),
            ("Al2O3 (sapphire)", "Al2O3 (sapphire)", 0, 13_000),
        )
    ),
    angle_range_deg=(0.0, 80.0),
    d_range_nm=(50.0, 350.0),
    h_range_nm=(80.0, 600.0),
    p_range_nm=(200.0, 600.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenGenericOnnxEvaluator:
    contract: GenericOnnxRouteContract
    model_path: str
    external_data_path: str
    source_pt_path: str
    session: Any = field(repr=False, compare=False)
    input_name: str = "input"
    output_name: str = "spectrum"

    def predict_spectra(
        self,
        geometries: object,
        *,
        material: str,
        substrate: str,
        polarization: str,
        angle_deg: float,
    ) -> np.ndarray | None:
        if self.contract.context_issue(material, substrate, polarization, angle_deg):
            return None
        if self.contract.geometry_issue(geometries):
            return None
        values = np.asarray(geometries, dtype=np.float32)
        pol_code = 0.0 if str(polarization).startswith("TE") else 1.0
        mat_code = float(self.contract.materials.index(material))
        sub_code = float(self.contract.substrates.index(substrate))
        model_input = np.column_stack((
            (values[:, 0] - 50.0) / 300.0,
            (values[:, 1] - 80.0) / 520.0,
            (values[:, 2] - 200.0) / 400.0,
            np.full(len(values), float(angle_deg) / 80.0),
            np.full(len(values), pol_code),
            np.full(len(values), mat_code),
            np.full(len(values), sub_code),
        )).astype(np.float32, copy=False)
        outputs = self.session.run(
            [self.output_name], {self.input_name: model_input})
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 1:
            return None
        spectra = np.asarray(outputs[0], dtype=float)
        if spectra.shape != (len(values), len(WL)):
            return None
        if not np.all(np.isfinite(spectra)):
            return None
        if np.any(spectra < 0.0) or np.any(spectra > 1.0):
            return None
        return spectra


@dataclass(frozen=True)
class GenericEvaluatorLoad:
    status: str
    reason: str
    evaluator: FrozenGenericOnnxEvaluator | None = field(
        default=None, repr=False, compare=False)
    evidence: ConversionEvidence | None = None

    @property
    def loaded(self) -> bool:
        return (
            self.status == "loaded"
            and self.evaluator is not None
            and self.evidence is not None
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"证据 JSON 不可读：{path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"证据 JSON 根节点不是对象：{path.name}")
    return value


def _canonical_json_sha256(value: object) -> str:
    payload = (json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_conversion_evidence(
    project_root: str | Path,
    contract: GenericOnnxRouteContract = GENERIC_ONNX_ROUTE,
) -> tuple[ConversionEvidence | None, str]:
    root = Path(project_root).resolve()
    protocol_path = (root / contract.conversion_protocol_relative_path).resolve()
    result_path = (root / contract.conversion_result_relative_path).resolve()
    if not protocol_path.is_file() or not result_path.is_file():
        return None, "PT→ONNX 版本化协议或结果证据缺失"
    try:
        protocol = _read_json(protocol_path)
        result = _read_json(result_path)
        protocol_hash = _canonical_json_sha256(protocol)
        result_hash = _canonical_json_sha256(result)
        if protocol_hash != contract.conversion_protocol_sha256:
            return None, "PT→ONNX 版本化协议哈希不匹配"
        if result_hash != contract.conversion_result_sha256:
            return None, "PT→ONNX 版本化结果哈希不匹配"
        if protocol.get("schema_version") != 1 or result.get("schema_version") != 1:
            raise ValueError("证据 schema_version 不受支持")
        if protocol.get("protocol_id") != result.get("protocol_id"):
            raise ValueError("协议与结果 protocol_id 不一致")
        if result.get("protocol_sha256") != protocol_hash:
            raise ValueError("结果未绑定当前协议哈希")
        if result.get("passed") is not True:
            raise ValueError("转换一致性结果未通过")
        if (
            protocol.get("training_commit") != contract.training_commit
            or result.get("training_commit") != contract.training_commit
            or protocol.get("conversion_commit") != contract.conversion_commit
            or result.get("conversion_commit") != contract.conversion_commit
        ):
            raise ValueError("训练或转换提交不匹配冻结合同")
        for key, expected_path, expected_hash in (
            ("source_pt", contract.source_pt_relative_path, contract.source_pt_sha256),
            ("onnx_graph", contract.model_relative_path, contract.model_sha256),
            ("onnx_external_data", contract.external_data_relative_path, contract.external_data_sha256),
        ):
            protocol_record = protocol["artifacts"][key]
            result_record = result["artifacts"][key]
            if protocol_record != {"path": expected_path, "sha256": expected_hash}:
                raise ValueError(f"协议 {key} 未绑定冻结 artifact")
            if result_record != protocol_record:
                raise ValueError(f"结果 {key} 未绑定协议 artifact")
        script_record = result["artifacts"]["audit_script"]
        script_path = (root / script_record["path"]).resolve()
        if not script_path.is_file() or _sha256(script_path) != script_record["sha256"]:
            raise ValueError("永久审计脚本缺失或哈希不匹配")
        sampling_protocol = protocol["sampling"]
        sampling_result = result["sampling"]
        if sampling_result["sample_count"] != sampling_protocol["sample_count"]:
            raise ValueError("结果 sample_count 与协议不一致")
        if sampling_result["input_shape"] != sampling_protocol["input_shape"]:
            raise ValueError("结果 input_shape 与协议不一致")
        if sampling_result["dtype"] != sampling_protocol["dtype"]:
            raise ValueError("结果 dtype 与协议不一致")
        if sampling_result["input_sha256"] != sampling_protocol["input_sha256"]:
            raise ValueError("结果 input SHA256 与协议不一致")
        inference_result = result["inference"]
        expected_output_shape = protocol["inference"]["output_shape"]
        if (
            inference_result["pt_output_shape"] != expected_output_shape
            or inference_result["onnx_output_shape"] != expected_output_shape
            or inference_result["output_dtype"] != protocol["inference"]["output_dtype"]
        ):
            raise ValueError("结果输出 shape/dtype 与协议不一致")
        metric = result["metric"]
        max_abs = float(metric["value"])
        tolerance = float(metric["tolerance"])
        if (
            metric["name"] != "max_abs"
            or tolerance != float(protocol["acceptance"]["max_abs_tolerance"])
            or not math.isfinite(max_abs)
            or max_abs < 0.0
            or max_abs > tolerance
        ):
            raise ValueError("转换一致性 metric 无效或未过阈值")
        evidence = ConversionEvidence(
            protocol_path=contract.conversion_protocol_relative_path,
            result_path=contract.conversion_result_relative_path,
            protocol_sha256=protocol_hash,
            result_sha256=result_hash,
            audit_script_sha256=script_record["sha256"],
            sample_count=int(sampling_result["sample_count"]),
            input_sha256=str(sampling_result["input_sha256"]),
            max_abs_diff=max_abs,
            tolerance=tolerance,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"PT→ONNX 版本化证据无效：{exc}"
    return evidence, ""


def freeze_generic_onnx_evaluator(
    project_root: str | Path,
    *,
    session_factory: Callable[..., Any] | None = None,
    contract: GenericOnnxRouteContract = GENERIC_ONNX_ROUTE,
) -> GenericEvaluatorLoad:
    """Load the exact local generic ONNX bundle; never download or auto-route."""
    root = Path(project_root).resolve()
    evidence, evidence_error = load_conversion_evidence(root, contract)
    if evidence is None:
        return GenericEvaluatorLoad("unavailable", evidence_error)
    model_path = (root / contract.model_relative_path).resolve()
    data_path = (root / contract.external_data_relative_path).resolve()
    source_pt_path = (root / contract.source_pt_relative_path).resolve()
    if not model_path.is_file() or not data_path.is_file() or not source_pt_path.is_file():
        return GenericEvaluatorLoad(
            "unavailable", "generic ONNX 主文件、外部权重或源 PT 文件缺失")
    if _sha256(model_path) != contract.model_sha256:
        return GenericEvaluatorLoad("unavailable", "generic ONNX 主文件哈希不匹配冻结合同")
    if _sha256(data_path) != contract.external_data_sha256:
        return GenericEvaluatorLoad("unavailable", "generic ONNX 外部权重哈希不匹配冻结合同")
    if _sha256(source_pt_path) != contract.source_pt_sha256:
        return GenericEvaluatorLoad("unavailable", "generic ONNX 源 PT 哈希不匹配冻结合同")
    try:
        if session_factory is None:
            import onnxruntime as ort
            session_factory = ort.InferenceSession
        session = session_factory(str(model_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "input" or inputs[0].shape[-1] != 7:
            return GenericEvaluatorLoad("unavailable", "generic ONNX 输入合同不是 input[...,7]")
        if len(outputs) != 1 or outputs[0].name != "spectrum" or outputs[0].shape[-1] != len(WL):
            return GenericEvaluatorLoad("unavailable", "generic ONNX 输出合同不是 spectrum[...,81]")
    except Exception as exc:
        return GenericEvaluatorLoad(
            "unavailable", f"generic ONNX 独立加载失败：{type(exc).__name__}")
    return GenericEvaluatorLoad(
        "loaded", "",
        FrozenGenericOnnxEvaluator(
            contract=contract,
            model_path=str(model_path),
            external_data_path=str(data_path),
            source_pt_path=str(source_pt_path),
            session=session,
        ),
        evidence,
    )


@dataclass(frozen=True)
class ModelDifferenceResult:
    status: str
    reason: str
    delta_e2000: tuple[float, ...] = ()
    fano_rgb: tuple[tuple[float, float, float], ...] = ()
    generic_rgb: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable"}:
            raise ValueError("unsupported model-difference status")
        if self.status == "unavailable":
            if self.delta_e2000 or self.fano_rgb or self.generic_rgb:
                raise ValueError("unavailable model-difference result cannot carry numbers")
            return
        metrics = np.asarray(self.delta_e2000, dtype=float)
        left = np.asarray(self.fano_rgb, dtype=float)
        right = np.asarray(self.generic_rgb, dtype=float)
        if metrics.ndim != 1 or metrics.size < 1:
            raise ValueError("available model-difference result requires metrics")
        if left.shape != (metrics.size, 3) or right.shape != (metrics.size, 3):
            raise ValueError("available model-difference RGB rows must match metrics")
        if (
            not np.all(np.isfinite(metrics)) or np.any(metrics < 0.0)
            or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
            or np.any(left < 0.0) or np.any(left > 1.0)
            or np.any(right < 0.0) or np.any(right > 1.0)
        ):
            raise ValueError("available model-difference numbers must be finite and valid")

    @property
    def available(self) -> bool:
        return self.status == "available" and bool(self.delta_e2000)


def _validated_spectra(value: object, n_samples: int) -> np.ndarray | None:
    try:
        spectra = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if spectra.shape != (n_samples, len(WL)) or not np.all(np.isfinite(spectra)):
        return None
    if np.any(spectra < 0.0) or np.any(spectra > 1.0):
        return None
    return spectra


def evaluate_fano_vs_generic(
    evaluator: FrozenGenericOnnxEvaluator,
    geometries: object,
    fano_spectra: object,
    *,
    material: str,
    substrate: str,
    polarization: str,
    angle_deg: float,
) -> ModelDifferenceResult:
    """Evaluate one explicit generic ONNX call against finite Fano spectra."""
    try:
        values = np.asarray(geometries, dtype=float)
    except (TypeError, ValueError):
        return ModelDifferenceResult("unavailable", "几何输入不可读")
    issue = evaluator.contract.context_issue(
        material, substrate, polarization, angle_deg)
    if issue:
        return ModelDifferenceResult("unavailable", issue)
    issue = evaluator.contract.geometry_issue(values)
    if issue:
        return ModelDifferenceResult("unavailable", issue)
    fano = _validated_spectra(fano_spectra, len(values))
    if fano is None:
        return ModelDifferenceResult("unavailable", "Fano 路线未返回有限 (N,81) 反射光谱")
    try:
        generic_raw = evaluator.predict_spectra(
            values, material=material, substrate=substrate,
            polarization=polarization, angle_deg=angle_deg)
    except Exception as exc:
        return ModelDifferenceResult(
            "unavailable", f"generic ONNX 独立调用失败：{type(exc).__name__}")
    generic = _validated_spectra(generic_raw, len(values))
    if generic is None:
        return ModelDifferenceResult(
            "unavailable", "generic ONNX 未返回有限 (N,81) 反射光谱")

    fano_rgb = np.asarray([spectrum_to_srgb(WL, row) for row in fano], dtype=float)
    generic_rgb = np.asarray(
        [spectrum_to_srgb(WL, row) for row in generic], dtype=float)
    if (
        fano_rgb.shape != (len(values), 3)
        or generic_rgb.shape != (len(values), 3)
        or not np.all(np.isfinite(fano_rgb))
        or not np.all(np.isfinite(generic_rgb))
    ):
        return ModelDifferenceResult("unavailable", "模型颜色转换未返回有限 (N,3) sRGB")
    metrics = np.asarray([
        delta_e2000(rgb_to_lab(left), rgb_to_lab(right))
        for left, right in zip(fano_rgb, generic_rgb)
    ], dtype=float)
    if metrics.shape != (len(values),) or not np.all(np.isfinite(metrics)):
        return ModelDifferenceResult("unavailable", "模型间 CIEDE2000 结果包含 NaN/Inf")
    return ModelDifferenceResult(
        "available", "",
        tuple(float(value) for value in metrics),
        tuple(tuple(float(channel) for channel in row) for row in fano_rgb),
        tuple(tuple(float(channel) for channel in row) for row in generic_rgb),
    )


def validate_model_difference_cache(value: object) -> ModelDifferenceResult | None:
    return value if isinstance(value, ModelDifferenceResult) else None
