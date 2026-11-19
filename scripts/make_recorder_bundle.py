"""Build the zip you send a teammate: the recorder agent, the web app, and your server address.

The teammate unzips it and double-clicks "Start Notetaker Recorder" (Mac: .command,
Windows: .cmd). The first start installs `uv` (a small Python manager) if needed, which then
fetches Python and the few packages the agent uses. No database, GPU or API keys involved:
recordings are uploaded to the server given here.

Usage: .venv/Scripts/python scripts/make_recorder_bundle.py [--server http://<tailscale-name>:8000]
       (default: this PC's Tailscale address, from `tailscale ip -4`)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"
NAME = "notetaker-recorder"

CODE = ["pipeline/__init__.py", "pipeline/recording.py", "pipeline/aec.py",
        "services/__init__.py", "services/recorder.py", "services/recorder_api.py",
        "services/calls.py", "services/agent.py", "services/security.py"]

PYPROJECT = """[project]
name = "notetaker-recorder"
version = "0.1.0"
description = "Records calls on this computer and sends them to the team's notetaker"
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "httpx>=0.27",
    "numpy>=2.0",
    "soundfile>=0.12",
    "soundcard>=0.4.6",
    "python-dotenv>=1.0",
]

[tool.uv]
package = false
"""

MAC = """#!/bin/bash
# Notetaker recorder (macOS). Double-click to start; close this window to stop.
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "First start: installing uv (Python manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Could not install uv."; read -r; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
export NOTETAKER_SERVER="$(cat server.txt)"
export NOTETAKER_WEB="$(pwd)/web"
echo "Notetaker recorder: http://localhost:8765  (server: $NOTETAKER_SERVER)"
(sleep 4; open "http://localhost:8765") &
uv run --quiet python -m services.agent
echo "The recorder stopped. Press Return to close."; read -r
"""

WIN = """@echo off
rem Notetaker recorder (Windows). Double-click to start; close this window to stop.
cd /d "%~dp0"
where uv >nul 2>nul || (
  echo First start: installing uv (Python manager)...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\\.local\\bin;%PATH%"
)
set /p NOTETAKER_SERVER=<server.txt
set "NOTETAKER_WEB=%~dp0web"
echo Notetaker recorder: http://localhost:8765  (server: %NOTETAKER_SERVER%)
start "" cmd /c "timeout /t 4 >nul & start http://localhost:8765"
uv run --quiet python -m services.agent
pause
"""


def tailscale_address() -> str | None:
    for exe in ("tailscale", r"C:\\Program Files\\Tailscale\\tailscale.exe"):
        try:
            out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=10,
                                 check=True).stdout.split()
            if out:
                return f"http://{out[0]}:8000"
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", help="the notetaker server as the teammate reaches it")
    p.add_argument("--out", type=Path, default=OUT, help="folder for the zip (default dist/)")
    args = p.parse_args()
    out_dir = args.out
    server = args.server or tailscale_address()
    if not server:
        sys.exit("No --server given and Tailscale isn't running here. Install Tailscale "
                 "(docs/teammate-setup.md) or pass --server http://<address>:8000.")
    web = ROOT / "apps" / "web" / "dist"
    subprocess.run("npm run build", cwd=ROOT / "apps" / "web", shell=True, check=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"{NAME}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        def add(arc: str, data: bytes | str, executable: bool = False) -> None:
            info = zipfile.ZipInfo(f"{NAME}/{arc}", date_time=time.localtime()[:6])
            info.create_system = 3          # Unix, so macOS keeps the permissions below
            info.external_attr = ((0o755 if executable else 0o644) | 0o100000) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data.encode() if isinstance(data, str) else data)

        for rel in CODE:
            add(rel, (ROOT / rel).read_bytes())
        for f in web.rglob("*"):
            if f.is_file():
                add(f"web/{f.relative_to(web).as_posix()}", f.read_bytes())
        add("pyproject.toml", PYPROJECT)
        add("server.txt", server + "\n")
        add("Start Notetaker Recorder.command", MAC.replace("\r\n", "\n"), executable=True)
        add("Start Notetaker Recorder.cmd", WIN.replace("\n", "\r\n"))
        add("SETUP.md", (ROOT / "docs" / "teammate-setup.md").read_text(encoding="utf-8"))
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB), server {server}")
    shutil.rmtree(out_dir / NAME, ignore_errors=True)


if __name__ == "__main__":
    main()
