# Start the notetaker server on Windows: database (Docker), API (also serves the web app),
# and GPU worker. Each opens in its own window so its log stays visible (and is also written
# to artifacts\logs\). Close a window to stop it.
#
# The API listens on all interfaces so teammates can reach it over Tailscale; it only
# answers this PC and Tailscale addresses (services/api.py). See docs/teammate-setup.md.
#
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File scripts\start.ps1 [-Dev] [-NoBrowser]
#   -Dev        also start the Vite dev server (hot reload) on http://localhost:5173
#   -NoBrowser  don't open the browser (used by the auto-start task, scripts\install_autostart.ps1)

param([switch]$Dev, [switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Docker Desktop may still be starting (e.g. right after login): start it and wait.
$ErrorActionPreference = "Continue"
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $dd = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Write-Host "Starting Docker Desktop..."; Start-Process $dd }
    for ($i = 0; $i -lt 60; $i++) { Start-Sleep -Seconds 3; docker info *> $null; if ($LASTEXITCODE -eq 0) { break } }
}
$ErrorActionPreference = "Stop"
Write-Host "Starting Postgres (Docker, port 5434)..."
docker compose -f infra/docker-compose.yml up -d | Out-Null

# The local notes model (NOTES_LLM=local) needs Ollama running.
try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/tags -TimeoutSec 3 | Out-Null }
catch {
    $ol = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (Test-Path $ol) { Write-Host "Starting Ollama..."; Start-Process $ol }
}

$web = Join-Path $root "apps\web"
$dist = Join-Path $web "dist\index.html"
$newest = Get-ChildItem (Join-Path $web "src") -Recurse -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not (Test-Path $dist) -or $newest.LastWriteTime -gt (Get-Item $dist).LastWriteTime) {
    Write-Host "Building the web app..."
    Push-Location $web; npm run build | Out-Null; Pop-Location
}

$py = Join-Path $root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force (Join-Path $root "artifacts\logs") | Out-Null
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root'; `$host.UI.RawUI.WindowTitle='notetaker API'; & '$py' -m uvicorn services.api:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 3 2>&1 | Tee-Object -FilePath artifacts\logs\api.log -Append"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root'; `$host.UI.RawUI.WindowTitle='notetaker worker (GPU)'; & '$py' -u -m services.worker 2>&1 | Tee-Object -FilePath artifacts\logs\worker.log -Append"
$url = "http://localhost:8000"
if ($Dev) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command",
        "Set-Location '$web'; `$host.UI.RawUI.WindowTitle='notetaker web (dev)'; npx vite --port 5173"
    $url = "http://localhost:5173"
}

Start-Sleep -Seconds 4
Write-Host "Open $url"
if (-not $NoBrowser) { Start-Process $url }
