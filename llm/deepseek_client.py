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


# --- Auto-load DEEPSEEK_API_KEY ---
try:
    import streamlit as _st
    if hasattr(_st, 'secrets') and 'DEEPSEEK_API_KEY' in _st.secrets:
        os.environ['DEEPSEEK_API_KEY'] = _st.secrets['DEEPSEEK_API_KEY']
except Exception:
    pass

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
                        if _val and len(_val) > 10:
                            os.environ["DEEPSEEK_API_KEY"] = _val
                            break
            if os.environ.get("DEEPSEEK_API_KEY"):
                break
    if os.environ.get("DEEPSEEK_API_KEY"):
        break


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

_CACHE = {}
_CACHE_MAX = 50
_CACHE_TTL = 300


def _cache_key(prompt, system_prompt, model, temperature):
    raw = f"{prompt}|{system_prompt}|{model}|{temperature}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_api_key():
    return os.environ.get("DEEPSEEK_API_KEY", "")


def chat(prompt, system_prompt="", model="deepseek-chat",
         temperature=0.7, max_tokens=256):
    """Call DeepSeek Chat API with LRU caching."""
    api_key = _get_api_key()
    if not api_key:
        return "[\u9519\u8bef] \u672a\u8bbe\u7f6e DEEPSEEK_API_KEY\u3002\u8bf7\u5728 .env \u6587\u4ef6\u4e2d\u914d\u7f6e"

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
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result = data["choices"][0]["message"]["content"]
            if len(_CACHE) >= _CACHE_MAX:
                oldest_key = min(_CACHE, key=lambda k: _CACHE[k][1])
                del _CACHE[oldest_key]
            _CACHE[ck] = (result, now)
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"[API \u9519\u8bef {e.code}] {body[:300]}"
    except Exception as e:
        return f"[\u7f51\u7edc\u9519\u8bef] {str(e)}"


def analyze_color(hex_color, params, spectrum_info=""):
    """Analyze metasurface color result with AI insights."""
    system = (
        "\u4f60\u662f\u8d85\u8868\u9762\u5149\u5b66\u4e13\u5bb6\uff0c\u7cbe\u901a\u7eb3\u7c73\u5149\u5b50\u5b66\u3001\u7c73\u6c0f\u5171\u632f\u3001\u7ed3\u6784\u8272\u7684\u539f\u7406\u3002"
        "\u7528\u6237\u4f1a\u7ed9\u4f60\u8d85\u8868\u9762\u7ed3\u6784\u8272\u7684\u8bbe\u8ba1\u53c2\u6570\u548c\u989c\u8272\u7ed3\u679c\uff0c"
        "\u8bf7\u4f60\u7528\u901a\u4fd7\u6613\u61c2\u7684\u4e2d\u6587\u89e3\u91ca\u989c\u8272\u4ea7\u751f\u7684\u7269\u7406\u539f\u56e0\uff0c"
        "\u5e76\u7ed9\u51fa\u4f18\u5316\u5efa\u8bae\u3002\u56de\u7b54\u63a7\u5236\u5728200\u5b57\u4ee5\u5185\uff0c\u52062-3\u4e2a\u8981\u70b9\u3002"
    )

    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    spec_line = f"\u5149\u8c31\u5206\u6790: {spectrum_info}\\n" if spectrum_info else ""
    prompt = (
        f"\u8d85\u8868\u9762\u7ed3\u6784\u8272\u53c2\u6570: {param_str}\\n"
        f"\u5f97\u5230\u7684\u989c\u8272: {hex_color}\\n"
        f"{spec_line}"
        f"\u8bf7\u7b80\u8981\u5206\u6790\u8fd9\u4e2a\u989c\u8272\u7684\u7269\u7406\u6765\u6e90\u5e76\u7ed9\u51fa\u4f18\u5316\u5efa\u8bae\u3002"
    )
    return chat(prompt, system_prompt=system, max_tokens=512)


def suggest_params(target_color, material, current_params):
    """Suggest parameter adjustments to approach a target color."""
    system = (
        "\u4f60\u662f\u8d85\u8868\u9762\u9006\u8bbe\u8ba1\u4e13\u5bb6\u3002\u6839\u636e\u76ee\u6807\u989c\u8272\u548c\u5f53\u524d\u53c2\u6570\uff0c"
        "\u7ed9\u51fa\u8c03\u6574\u5efa\u8bae\uff08\u589e\u5927/\u51cf\u5c0f\u76f4\u5f84D\u3001\u9ad8\u5ea6H\u3001\u5468\u671fP\u7684\u65b9\u5411\u548c\u5e45\u5ea6\uff09\u3002"
        "\u56de\u7b54\u7b80\u6d01\uff0c\u5206\u70b9\u5217\u51fa\uff0c\u6bcf\u70b9\u4e00\u884c\u3002"
    )

    cur = ", ".join(f"{k}={v}" for k, v in current_params.items())
    prompt = (
        f"\u6750\u6599: {material}\\n"
        f"\u76ee\u6807\u989c\u8272: {target_color}\\n"
        f"\u5f53\u524d\u53c2\u6570: {cur}\\n"
        f"\u8bf7\u5efa\u8bae\u5982\u4f55\u8c03\u6574\u53c2\u6570\u6765\u903c\u8fd1\u76ee\u6807\u989c\u8272\u3002"
    )
    return chat(prompt, system_prompt=system, max_tokens=512)