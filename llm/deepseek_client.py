"""DeepSeek API client for metasurface color analysis.

Uses DeepSeek Chat API (OpenAI-compatible) for intelligent color insights.
API key should be set as environment variable DEEPSEEK_API_KEY.
"""

import os
import json
import hashlib
import time
import urllib.request
import urllib.error


# --- Auto-load API key ---
# Priority: 1) st.secrets (Streamlit Cloud)  2) .env file (local)  3) env var
try:
    import streamlit as _st
    if hasattr(_st, 'secrets') and 'DEEPSEEK_API_KEY' in _st.secrets:
        os.environ['DEEPSEEK_API_KEY'] = _st.secrets['DEEPSEEK_API_KEY']
except Exception:
    pass

# Fallback: .env file (local)
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
                    if _line.startswith("DEEPSEEK_API_KEY=") and not _line.startswith("#"):
                        _val = _line.split("=", 1)[1].strip()
                        if _val and "粘贴" not in _val and "你的key" not in _val:
                            os.environ["DEEPSEEK_API_KEY"] = _val
                            break
            if os.environ.get("DEEPSEEK_API_KEY"):
                break
    if os.environ.get("DEEPSEEK_API_KEY"):
        break


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# --- LRU response cache (max 50 entries, 30s TTL) ---
_CACHE = {}
_CACHE_MAX = 50
_CACHE_TTL = 30  # seconds


def _cache_key(prompt, system_prompt, model, temperature):
    raw = f"{prompt}|{system_prompt}|{model}|{temperature}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_api_key() -> str:
    """Get DeepSeek API key from environment."""
    return os.environ.get("DEEPSEEK_API_KEY", "")


def chat(prompt: str, system_prompt: str = "", model: str = "deepseek-chat",
         temperature: float = 0.7, max_tokens: int = 256) -> str:
    """Call DeepSeek Chat API with LRU caching.

    Args:
        prompt: User message.
        system_prompt: System role instruction.
        model: Model name (deepseek-chat or deepseek-reasoner).
        temperature: Sampling temperature.
        max_tokens: Max output tokens.

    Returns:
        Model response text, or error message on failure.
    """
    api_key = _get_api_key()
    if not api_key:
        return "[错误] 未设置 DEEPSEEK_API_KEY 环境变量。请在终端运行: set DEEPSEEK_API_KEY=你的key"

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

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(DEEPSEEK_API_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result = data["choices"][0]["message"]["content"]
            # Update cache
            if len(_CACHE) >= _CACHE_MAX:
                oldest_key = min(_CACHE, key=lambda k: _CACHE[k][1])
                del _CACHE[oldest_key]
            _CACHE[ck] = (result, now)
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"[API 错误 {e.code}] {body[:300]}"
    except Exception as e:
        return f"[网络错误] {str(e)}"


def analyze_color(hex_color: str, params: dict, spectrum_info: str = "") -> str:
    """Analyze current metasurface color result with AI insights.

    Args:
        hex_color: Hex color code (e.g. "#80c8ff").
        params: Dict with keys like D, H, P, material, substrate, etc.
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


def suggest_params(target_color: str, material: str, current_params: dict) -> str:
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
