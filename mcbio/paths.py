"""Where mcbio finds its bundled files and where it writes data.

Two ways to run:

* **From source** (``python -m mcbio``): profiles and models live next to the package,
  and session logs go wherever ``--log-dir`` points (the launchers use ``sessions/``).
* **As the packaged Mac app** (PyInstaller, ``sys.frozen``): profiles and models are
  unpacked into the bundle (``sys._MEIPASS``), which is read-only, so everything the
  app writes goes to ``~/Documents/Minecraft Biogaming/`` instead.

Calibration is always ``~/.mcbio/calibration.json`` (see ``calibration.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "Minecraft Biogaming"
SOURCE_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """True when running inside the packaged app (PyInstaller sets ``sys.frozen``)."""
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    """Folder that holds ``profiles/`` and ``models/`` (read-only inside the app)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return SOURCE_ROOT


def profiles_dir() -> Path:
    return resource_dir() / "profiles"


def models_dir() -> Path:
    return resource_dir() / "models"


def data_dir(create: bool = True) -> Path:
    """Writable folder for session logs and the app's log file.

    Source runs keep using the project folder; the packaged app uses
    ``~/Documents/Minecraft Biogaming`` so users can find their recordings in Finder.
    """
    path = Path.home() / "Documents" / APP_NAME if is_frozen() else SOURCE_ROOT
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def app_version() -> str:
    """The app's version (from the bundle's Info.plist), or "source" when run from source."""
    if not is_frozen():
        return "source"
    import plistlib
    try:
        plist = Path(sys.executable).resolve().parents[1] / "Info.plist"   # Contents/MacOS/x -> Contents/
        with open(plist, "rb") as fh:
            return str(plistlib.load(fh).get("CFBundleShortVersionString", "unknown"))
    except (OSError, ValueError, IndexError):
        return "unknown"
