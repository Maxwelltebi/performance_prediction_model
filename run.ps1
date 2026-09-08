# Start the Student Performance app on http://127.0.0.1:8000
$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path $PSScriptRoot "backend")
& (Join-Path $PSScriptRoot "venv\Scripts\python.exe") -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
