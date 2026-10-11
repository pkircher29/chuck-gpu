# Build dist\chuck-gpu\ (chuck-gpu.exe + chuck-gpu-agent.exe) and dist\chuck-gpu-windows.zip.
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Push-Location $PSScriptRoot
try {
    # build from the source tree; don't reinstall the package (its chuck-gpu.exe may be in use)
    & $py -m pip install -q pyinstaller "modal>=1.6" "pystray>=0.19" "pillow>=10"
    $env:PYTHONPATH = $root
    & $py -c "from chuck_gpu.agent import save_ico; save_ico('chuck-gpu.ico')"
    if ($LASTEXITCODE) { throw "icon generation failed" }
    & $py -m PyInstaller --noconfirm --clean --distpath "$root\dist" --workpath "$root\build" chuck-gpu.spec
    if ($LASTEXITCODE) { throw "PyInstaller failed" }
    Copy-Item install.ps1 "$root\dist\chuck-gpu\install.ps1" -Force
    $zip = "$root\dist\chuck-gpu-windows.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Compress-Archive -Path "$root\dist\chuck-gpu\*" -DestinationPath $zip
    Write-Host "built $root\dist\chuck-gpu  and  $zip"
} finally {
    Pop-Location
}
