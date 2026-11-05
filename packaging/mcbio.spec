# -*- mode: python -*-
# PyInstaller recipe for "Minecraft Biogaming.app" (macOS). Build from the repo root:
#
#     python scripts/download_models.py          # models/*.task must exist
#     pyinstaller --noconfirm packaging/mcbio.spec
#     bash packaging/make_dmg.sh                 # -> dist/Minecraft-Biogaming-<ver>-mac-<arch>.dmg
#
# The app bundles Python, MediaPipe, OpenCV, pynput/Quartz, the two models and the
# profiles. It is ad-hoc signed only (no Apple Developer ID), see packaging/HOW_TO_OPEN.txt.
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
NAME = "Minecraft Biogaming"
VERSION = os.environ.get("MCBIO_VERSION", "0.1.0")

for model in ("hand_landmarker.task", "face_landmarker.task"):
    if not os.path.isfile(os.path.join(ROOT, "models", model)):
        raise SystemExit(f"models/{model} missing - run: python scripts/download_models.py")

datas = [
    (os.path.join(ROOT, "profiles", "*.yaml"), "profiles"),
    (os.path.join(ROOT, "models", "*.task"), "models"),
]
binaries = []
hiddenimports = collect_submodules("mcbio") + collect_submodules("pynput")
# MediaPipe loads native libraries and graph/model data at runtime: take all of it.
_d, _b, _h = collect_all("mediapipe")
datas += _d
binaries += _b
hiddenimports += _h

a = Analysis(
    [os.path.join(SPECPATH, "app_entry.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "tkinter", "IPython", "notebook", "torch", "tensorflow", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=False,
    argv_emulation=False,
    codesign_identity=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME)
app = BUNDLE(
    coll,
    name=f"{NAME}.app",
    icon=None,
    bundle_identifier="io.github.arthurhsia.mcbio",
    version=VERSION,
    info_plist={
        "CFBundleDisplayName": NAME,
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.games",
        "NSHighResolutionCapable": True,
        # Shown in the macOS camera prompt. Without it macOS kills the app on camera access.
        "NSCameraUsageDescription": (
            "Minecraft Biogaming uses the camera to track your face and hands so you can "
            "control Minecraft. Video never leaves your Mac."),
    },
)
