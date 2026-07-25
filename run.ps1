$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "尚未创建虚拟环境，请先运行 .\setup.ps1"
}

Set-Location $ProjectRoot
& $VenvPython -m streamlit run app.py

