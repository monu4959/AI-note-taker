"""Is a Zoom or Teams call happening right now? Read from Windows' microphone privacy record.

Windows logs, per app, when it last started and stopped using the microphone (the data
behind the "app is using your microphone" indicator). An app whose last use has a start but
no stop is using the mic now. Zoom and Teams open the mic only for a call (it stays open
while you are muted), so "Zoom has the mic" means "you are in a Zoom call".

No extra software and no permission prompt: these are the current user's own registry keys.

macOS (teammates' recorder agent; untested here, no Mac available): Zoom runs a helper
process, CptHost, only while you are in a meeting. Teams has no such process, so "Teams is
running and some app is using the default mic" (CoreAudio's IsRunningSomewhere) counts as a
Teams call. While this recorder itself uses the mic, that signal is always on, so on a Mac a
Teams call can be detected but its end cannot (Zoom's can).
"""

from __future__ import annotations

import sys

BASE = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"

# desktop apps, by executable name (NonPackaged keys are the exe path with \ -> #)
EXES = {"zoom.exe": "Zoom", "teams.exe": "Teams", "ms-teams.exe": "Teams"}
# Store-packaged apps, by package family prefix (the new Teams client is packaged)
PACKAGES = {"msteams_": "Teams", "zoom.zoom": "Zoom"}


def _in_use(key) -> bool:
    import winreg

    try:
        start = winreg.QueryValueEx(key, "LastUsedTimeStart")[0]
        stop = winreg.QueryValueEx(key, "LastUsedTimeStop")[0]
    except OSError:
        return False
    return bool(start) and not stop


def _subkeys(key):
    import winreg

    i = 0
    while True:
        try:
            yield winreg.EnumKey(key, i)
        except OSError:
            return
        i += 1


def _mac_mic_in_use() -> bool:
    import ctypes
    import ctypes.util

    ca = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio"))

    class Address(ctypes.Structure):
        _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32),
                    ("element", ctypes.c_uint32)]

    def u32(obj: int, selector: str) -> int | None:
        addr = Address(int.from_bytes(selector.encode(), "big"),
                       int.from_bytes(b"glob", "big"), 0)
        size, val = ctypes.c_uint32(4), ctypes.c_uint32(0)
        status = ca.AudioObjectGetPropertyData(ctypes.c_uint32(obj), ctypes.byref(addr), 0,
                                               None, ctypes.byref(size), ctypes.byref(val))
        return val.value if status == 0 else None

    device = u32(1, "dIn ")            # kAudioObjectSystemObject -> default input device
    return bool(device and u32(device, "gone"))   # kAudioDevicePropertyDeviceIsRunningSomewhere


def _mac_calls() -> list[str]:
    import subprocess

    try:
        procs = set(subprocess.run(["ps", "-axco", "comm"], capture_output=True, text=True,
                                   timeout=5).stdout.splitlines())
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    if "CptHost" in procs:
        found.append("Zoom")
    if procs & {"MSTeams", "Microsoft Teams", "Microsoft Teams (work or school)"}:
        try:
            if _mac_mic_in_use():
                found.append("Teams")
        except OSError:
            pass
    return found


def active_calls() -> list[str]:
    """Names of call apps in a call now, e.g. ["Zoom"]. [] where not supported."""
    if sys.platform == "darwin":
        return _mac_calls()
    if sys.platform != "win32":
        return []
    import winreg

    found: set[str] = set()
    try:
        root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, BASE)
    except OSError:
        return []
    with root:
        for name in _subkeys(root):
            lower = name.lower()
            if lower == "nonpackaged":
                with winreg.OpenKey(root, name) as np_key:
                    for exe_path in _subkeys(np_key):
                        app = EXES.get(exe_path.rsplit("#", 1)[-1].lower())
                        if app and app not in found:
                            with winreg.OpenKey(np_key, exe_path) as k:
                                if _in_use(k):
                                    found.add(app)
                continue
            app = next((a for prefix, a in PACKAGES.items() if lower.startswith(prefix)), None)
            if app and app not in found:
                with winreg.OpenKey(root, name) as k:
                    if _in_use(k):
                        found.add(app)
    return sorted(found)


if __name__ == "__main__":
    print(active_calls() or "no call app is using the microphone")
