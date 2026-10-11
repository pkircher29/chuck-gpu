# Install mgpu for the current user (no admin needed):
#   * copies this folder to %LOCALAPPDATA%\Programs\mgpu
#   * puts it on your PATH, so `mgpu` works in any new terminal
#   * starts the tray agent, which registers itself to start with Windows
# Run from the unzipped folder:
#   powershell -ExecutionPolicy Bypass -File install.ps1            (install / update)
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall
param([switch]$Uninstall)
$ErrorActionPreference = "Stop"
$dest = Join-Path $env:LOCALAPPDATA "Programs\mgpu"

# stop a running agent so its files can be replaced
Get-Process mgpu-agent -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

$path = [Environment]::GetEnvironmentVariable("Path", "User")
$parts = @($path -split ";" | Where-Object { $_ -and ($_.TrimEnd("\") -ne $dest) })

if ($Uninstall) {
    [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "User")
    Remove-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name mgpu -ErrorAction SilentlyContinue
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Write-Host "mgpu removed. Your Modal login (~\.modal.toml) and job history (~\.mgpu) were left in place."
    return
}

if ($PSScriptRoot.TrimEnd("\") -ne $dest) {
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item (Join-Path $PSScriptRoot "*") $dest -Recurse -Force
}
[Environment]::SetEnvironmentVariable("Path", (@($dest) + $parts) -join ";", "User")
Start-Process (Join-Path $dest "mgpu-agent.exe")
Write-Host "mgpu installed to $dest"
Write-Host "Look for the mgpu icon in the system tray; open a NEW terminal and run:  mgpu login"
