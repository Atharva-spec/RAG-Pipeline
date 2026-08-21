# Direct launcher for RAG Pipeline on Windows (PowerShell)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$PythonExe = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = Join-Path $ScriptDir "venv\bin\python.exe"
}

if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment not found. Run 'npm install' or 'python -m venv venv; .\venv\Scripts\pip install -r requirements.txt' first."
    exit 1
}

& $PythonExe (Join-Path $ScriptDir "main.py") $args
