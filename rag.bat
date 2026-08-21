@echo off
REM Direct launcher for RAG Pipeline on Windows (CMD / Batch)
setlocal

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%venv\bin\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%venv\bin\python.exe"
) else (
    echo [ERROR] Virtual environment not found.
    echo Run 'npm install' or 'python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt' first.
    exit /b 1
)

"%PYTHON_EXEC%" "%SCRIPT_DIR%main.py" %*
