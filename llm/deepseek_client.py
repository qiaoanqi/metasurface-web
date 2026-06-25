"""HF Inference API client (free tier) for metasurface color analysis.

Uses Hugging Face InferenceClient for intelligent color insights.
API key should be set as environment variable HF_TOKEN.
"""

import os
import json
import hashlib
import time


# --- Auto-load HF_TOKEN ---
# Priority: 1) st.secrets  2) .env file  3) env var
try:
    import streamlit as _st
    if hasattr(_st, "secrets") and "HF_TOKEN" in _st.secrets:
        os.environ["HF_TOKEN"] = _st.secrets["HF_TOKEN"]
except Exception:
    pass

# Fallback: .env file
for _base in [
    os.path.dirname(os.path.abspath(__file__)),
    os.getcwd(),
]:
    for _name in ["../.env", ".env"]:
        _env_path = os.path.normpath(os.path.join(_base, _name))
        if os.path.exists(_env_path):
            with open(_env_path, "r", encoding="utf-8-sig") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line.startswith("HF_TOKEN=") and not _line.startswith("#"):
                        _val = _line.split("=", 1)[1].strip()
                        if _val and len(_val) > 10:
                            os.environ["HF_TOKEN"] = _val
                            break
            if os.environ.get("HF_TOKEN"):
                break
    if os.environ.get("HF_TOKEN"):
        break


# --- LRU response cache ---
_CACHE = {}
_CACHE_MAX = 50
_CACHE_TTL = 300


def _cache_key(prompt, system_prompt, model, temperature):
    raw = f"{prompt}|{system_prompt}|{model}|{temperature}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_api_key():
    return os.environ.get("HF_TOKEN", "")


def _get_client():
    """Get or create an InferenceClient instance."""
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        from huggingface_hub import InferenceClient
        return InferenceClient(token=api_key)
    except ImportError:
        return None


def chat(prompt, system_prompt="", model="Qwen/Qwen2.5-7B-Instruct",
         temperature=0.7, max_tokens=256):
    """Call HF Inference API with LRU caching.

    Args:
        prompt: User message.
        system_prompt: System role instruction.
        model: HF model ID (e.g. Qwen/Qwen2.5-7B-Instruct).
        temperature: Sampling temperature.
        max_tokens: Max output tokens.

    Returns:
        Model response text, or error message on failure.
    """
    api_key = _get_api_key()
    if not api_key:
        return ("[错误] 未设置 HF_TOKEN 环境变量。\n"
                "请在项目根目录创建 .env 文件，写入 HF_TOKEN=你的HF令牌")

    client = _get_client()
    if client is None:
        return "[错误] 未安装 huggingface_hub 库。请运行: pip install huggingface_hub"

    # Check cache
    ck = _cache_key(prompt, system_prompt, model, temperature)
    now = time.time()
    if ck in _CACHE:
        cached_result, cached_time = _CACHE[ck]
        if now - cached_time < _CACHE_TTL:
            return cached_result

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content

        # Update cache
        if len(_CACHE) >= _CACHE_MAX:
            oldest_key = min(_CACHE, key=lambda k: _CACHE[k][1])
            del _CACHE[oldest_key]
        _CACHE[ck] = (result, now)
        return result

    except Exception as e:
        err_msg = str(e)
        # Truncate long error messages
        if len(err_msg) > 300:
            err_msg = err_msg[:300] + "..."
        return f"[API 错误] {err_msg}"


def analyze_color(hex_color, params, spectrum_info=""):
    """Analyze current metasurface color result with AI insights.

    Args:
        hex_color: Hex color code (e.g. "#80c8ff").
        params: Dict with keys like D, H, P, material, substrate.
        spectrum_info: Optional spectral analysis summary.

    Returns:
        AI analysis text in Chinese.
    """
    system = (
        "你是超表面光学专家，精通纳米光子学、米氏共振、结构色的原理。"
        "用户会给你超表面结构色的设计参数和颜色结果，"
        "请你用通俗易懂的中文解释颜色产生的物理原因，"
        "并给出优化建议。回答控制在200字以内，分2-3个要点。"
    )

    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    spec_line = f"光谱分析: {spectrum_info}\n" if spectrum_info else ""
    prompt = (
        f"超表面结构色参数: {param_str}\n"
        f"得到的颜色: {hex_color}\n"
        f"{spec_line}"
        f"请简要分析这个颜色的物理来源并给出优化建议。"
    )
    return chat(prompt, system_prompt=system, max_tokens=512)


def suggest_params(target_color, material, current_params):
    """Suggest parameter adjustments to approach a target color.

    Args:
        target_color: Target hex color.
        material: Material name.
        current_params: Current parameter dict.

    Returns:
        AI suggestion text.
    """
    system = (
        "你是超表面逆设计专家。根据目标颜色和当前参数，"
        "给出调整建议（增大/减小直径D、高度H、周期P的方向和幅度）。"
        "回答简洁，分点列出，每点一行。"
    )

    cur = ", ".join(f"{k}={v}" for k, v in current_params.items())
    prompt = (
        f"材料: {material}\n"
        f"目标颜色: {target_color}\n"
        f"当前参数: {cur}\n"
        f"请建议如何调整参数来逼近目标颜色。"
    )
    return chat(prompt, system_prompt=system, max_tokens=512)
