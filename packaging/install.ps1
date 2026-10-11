# Install chuck-gpu for the current user (no admin needed):
#   * copies this folder to %LOCALAPPDATA%\Programs\chuck-gpu
#   * puts it on your PATH, so `chuck-gpu` works in any new terminal
#   * starts the tray agent, which registers itself to start with Windows
# Run from the unzipped folder:
#   powershell -ExecutionPolicy Bypass -File install.ps1            (install / update)
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall
param([switch]$Uninstall)
$ErrorActionPreference = "Stop"
$dest = Join-Path $env:LOCALAPPDATA "Programs\chuck-gpu"

# stop a running agent so its files can be replaced
Get-Process chuck-gpu-agent -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

$path = [Environment]::GetEnvironmentVariable("Path", "User")
$parts = @($path -split ";" | Where-Object { $_ -and ($_.TrimEnd("\") -ne $dest) })

if ($Uninstall) {
    [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "User")
    Remove-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name chuck-gpu -ErrorAction SilentlyContinue
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Write-Host "chuck-gpu removed. Your Modal login (~\.modal.toml) and job history (~\.chuck_gpu) were left in place."
    return
}

if ($PSScriptRoot.TrimEnd("\") -ne $dest) {
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item (Join-Path $PSScriptRoot "*") $dest -Recurse -Force
}
[Environment]::SetEnvironmentVariable("Path", (@($dest) + $parts) -join ";", "User")
Start-Process (Join-Path $dest "chuck-gpu-agent.exe")
Write-Host "chuck-gpu installed to $dest"
Write-Host "Look for the chuck-gpu icon in the system tray; open a NEW terminal and run:  chuck-gpu login"
