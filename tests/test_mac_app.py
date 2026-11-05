"""Packaged-app support: bundle paths, start screen, permission helpers, focus notice."""
import sys
from pathlib import Path

import numpy as np
import pytest

from mcbio import launcher, macperms, paths


def test_source_paths_point_at_the_repo():
    assert not paths.is_frozen()
    assert (paths.profiles_dir() / "face_hand.yaml").is_file()
    assert paths.models_dir() == paths.SOURCE_ROOT / "models"
    assert paths.data_dir(create=False) == paths.SOURCE_ROOT


def test_frozen_paths_use_bundle_and_documents(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    assert paths.resource_dir() == tmp_path / "bundle"
    assert paths.profiles_dir() == tmp_path / "bundle" / "profiles"
    d = paths.data_dir()
    assert d == tmp_path / "home" / "Documents" / paths.APP_NAME and d.is_dir()


def test_controller_args_match_the_cli():
    from mcbio.app import config_from_args
    st = launcher.StartState(profile=1, camera=2)
    cfg = config_from_args(launcher.controller_args(st))
    assert cfg.profile.endswith("face_only.yaml") and Path(cfg.profile).is_file()
    assert cfg.camera == 2 and cfg.calibrate and cfg.backend == "pynput"
    assert cfg.log_dir.endswith("sessions") and cfg.hud


def test_every_start_screen_profile_loads():
    from mcbio.mapping import Profile
    for _, name, _ in launcher.PROFILES:
        Profile.load(paths.profiles_dir() / name)


@pytest.mark.parametrize("preview", [None, np.zeros((480, 640, 3), np.uint8)])
def test_start_screen_draws_and_buttons_hit(preview):
    pytest.importorskip("cv2")
    st = launcher.StartState(camera_ok=preview is not None,
                             perms={"accessibility": False, "input_monitoring": None},
                             message="x " * 120, message_is_error=True)
    img, buttons = launcher.draw_start_screen(preview, st)
    assert img.shape == (launcher.H, launcher.W, 3)
    names = {n for n, _ in buttons}
    assert {"start", "cam_prev", "cam_next", "profile_0", "profile_1", "profile_2"} <= names
    x0, y0, x1, y1 = dict(buttons)["start"]
    assert launcher.hit(buttons, (x0 + x1) // 2, (y0 + y1) // 2) == "start"
    assert launcher.hit(buttons, 1, 1) is None


def test_launcher_passes_cli_arguments_through():
    rc = launcher.main(["--synthetic", "--dry-run", "--no-hud", "--frames", "20", "--quiet"])
    assert rc == 0


def test_permission_helpers_never_raise():
    for fn in (macperms.accessibility, macperms.input_monitoring):
        assert fn() in (True, False, None)
    if sys.platform != "darwin":
        assert macperms.accessibility() is None and not macperms.open_settings("camera")
    assert not macperms.open_settings("nonsense")


def test_focus_change_sets_and_clears_hud_notice():
    from mcbio.app import App, AppConfig
    app = App(AppConfig(synthetic=True, backend="dry", hud=False, quiet=True))
    app._on_focus(False, "Finder")
    assert app.output_notice and "Minecraft" in app.output_notice
    app._on_focus(True, "minecraft")
    assert app.output_notice is None


def test_hud_draws_notice():
    pytest.importorskip("cv2")
    from mcbio.hud import HudState, draw_hud
    frame = np.zeros((720, 1280, 3), np.uint8)
    a = draw_hud(frame, None, None, state=HudState(status="LIVE"))
    b = draw_hud(frame, None, None, state=HudState(status="LIVE", notice="Click the Minecraft window"))
    assert (a != b).any()


def test_pyinstaller_spec_and_dmg_script_reference_real_files():
    root = paths.SOURCE_ROOT
    spec = (root / "packaging" / "mcbio.spec").read_text()
    assert "NSCameraUsageDescription" in spec and "app_entry.py" in spec
    assert (root / "packaging" / "app_entry.py").is_file()
    dmg = (root / "packaging" / "make_dmg.sh").read_text()
    assert "HOW_TO_OPEN.txt" in dmg and (root / "packaging" / "HOW_TO_OPEN.txt").is_file()
    for arch in ("apple-silicon", "intel"):
        assert (root / "packaging" / f"requirements-mac-{arch}.txt").is_file()
