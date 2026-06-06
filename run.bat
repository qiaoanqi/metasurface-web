@echo off
chcp 65001 >nul
title AI超表面结构色设计系统

echo ============================================
echo   AI超表面结构色设计系统 v3.0
echo   长沙理工大学 物电学院 光电2501
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: Install dependencies (first run)
if not exist ".deps_installed" (
    echo [提示] 首次运行，正在安装依赖库...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if %errorlevel% equ 0 (
        type nul > .deps_installed
        echo [完成] 依赖安装成功！
    ) else (
        echo [错误] 依赖安装失败，请检查网络连接后重试
        pause
        exit /b 1
    )
)

echo [启动] Streamlit 应用...
echo [地址] http://localhost:8501
echo.

streamlit run app.py --server.port 8501

pause
