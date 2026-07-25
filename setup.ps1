$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$InstalledPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path $VenvPython)) {
    Write-Host "正在创建 Python 虚拟环境 .venv ..."
    if (Test-Path $InstalledPython) {
        & $InstalledPython -m venv .venv
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    } else {
        throw "Python 3.12 was not found. Install Python and try again."
    }
}

Write-Host "正在安装项目依赖 ..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") (Join-Path $ProjectRoot ".env")
    Write-Host ""
    Write-Warning "已创建 .env，请在其中填写 SILICONFLOW_API_KEY。"
}

Write-Host ""
Write-Host "环境准备完成。配置 .env 后运行 .\run.ps1 启动。"

