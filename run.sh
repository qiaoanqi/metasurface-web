#!/bin/bash
# AI超表面结构色智能设计系统 - Linux/macOS一键运行脚本
echo "=== AI超表面结构色设计系统 v5.0 ==="
echo "安装依赖..."
pip install -r requirements.txt
echo "启动应用..."
streamlit run app.py --server.port 8501