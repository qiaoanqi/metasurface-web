#!/bin/bash
set -e
echo "============================================"
echo "  AI?????????? v5.0"
echo "  ?????? ???? ??2501"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[??] ??? Python3????? Python 3.10+"
    exit 1
fi

# Create virtual environment (first run)
if [ ! -f ".venv/bin/python" ]; then
    echo "[??] ?????????????..."
    python3 -m venv .venv
fi

# Activate and install deps (first run)
if [ ! -f ".deps_installed" ]; then
    echo "[??] ???????..."
    source .venv/bin/activate
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    touch .deps_installed
    echo "[??] ???????"
fi

echo "[??] Streamlit ??..."
echo "[??] http://localhost:8501"
echo ""

.venv/bin/streamlit run app.py --server.port 8501