@echo off
chcp 65001 >nul
title AI??????????

echo ============================================
echo   AI?????????? v5.0
echo   ?????? ???? ??2501
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [??] ??? Python????? Python 3.10+
    pause
    exit /b 1
)

:: Create virtual environment (first run)
if not exist ".venv\Scripts\python.exe" (
    echo [??] ?????????????...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [??] ????????
        pause
        exit /b 1
    )
)

:: Activate venv and install deps (first run)
if not exist ".deps_installed" (
    echo [??] ???????...
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if %errorlevel% equ 0 (
        type nul > .deps_installed
        echo [??] ???????
    ) else (
        echo [??] ?????????????????
        pause
        exit /b 1
    )
)

echo [??] Streamlit ??...
echo [??] http://localhost:8501
echo.

.venv\Scripts\streamlit run app.py --server.port 8501

pause