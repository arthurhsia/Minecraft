"""Java Edition option, Quit menu flag, pynput keyboard-layout cache, app version."""
import contextlib
import plistlib
import sys
import types

import pytest

from mcbio import launcher, macapp, paths
from mcbio.app import GAMES, App, AppConfig, config_from_args, make_backend


# ------------------------------------------------------------------ Java Edition
def test_game_flag_parses_and_defaults_to_education():
    assert config_from_args([]).game == "education"
    assert config_from_args(["--game", "java"]).game == "java"


def test_java_moves_sprint_off_ctrl():
    app = App(AppConfig(synthetic=True, backend="dry", hud=False, quiet=True, frames=3, game="java"))
    app.run()
    assert app.profile.bindings_keys["sprint"] == "r"
    edu = App(AppConfig(synthetic=True, backend="dry", hud=False, quiet=True, frames=3))
    edu.run()
    assert edu.profile.bindings_keys["sprint"] == "ctrl"


def test_java_focus_lock_accepts_the_java_window(monkeypatch):
    from mcbio import outputs
    front = {"name": "java"}
    monkeypatch.setattr(outputs, "frontmost_app_name", lambda: front["name"])
    monkeypatch.setattr(outputs, "PynputOutput", lambda keys=None: outputs.DryRunOutput(keys))
    gated = make_backend("pynput", also_targets=GAMES["java"]["also_targets"])
    assert isinstance(gated, outputs.FocusGatedOutput)
    assert any(t in "java" for t in gated.targets) and any(t in "minecraft" for t in gated.targets)
    edu = make_backend("pynput", also_targets=GAMES["education"]["also_targets"])
    assert not any(t in "java" for t in edu.targets)


def test_start_screen_game_buttons_and_args():
    pytest.importorskip("cv2")
    st = launcher.StartState(game=1)
    _, buttons = launcher.draw_start_screen(None, st)
    names = {n for n, _ in buttons}
    assert {"game_0", "game_1"} <= names
    args = launcher.controller_args(st)
    assert args[args.index("--game") + 1] == "java"
    assert config_from_args(args).game == "java"


# ------------------------------------------------------------------ Quit
def test_quit_request_stops_the_controller_loop():
    macapp.reset_quit()
    app = App(AppConfig(synthetic=True, backend="dry", hud=False, quiet=True, frames=500))
    macapp.request_quit()
    try:
        summary = app.run()
    finally:
        macapp.reset_quit()
    assert summary["frames"] == 0 and summary["released_cleanly"]


def test_menu_and_cache_are_noops_off_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert macapp.install_quit_menu() is False
    assert macapp.harden_pynput() is False


# ------------------------------------------------------------------ pynput keyboard-layout cache
def test_harden_pynput_reads_layout_once_and_reuses_it(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def real_context():
        calls.append(1)
        yield ("kbd-type", b"layout-bytes")

    util = types.ModuleType("pynput._util.darwin")
    util.keycode_context = real_context
    kb = types.ModuleType("pynput.keyboard._darwin")
    kb.keycode_context = real_context
    pkg = types.ModuleType("pynput")
    pkg._util = types.ModuleType("pynput._util")
    pkg.keyboard = types.ModuleType("pynput.keyboard")
    pkg._util.darwin, pkg.keyboard._darwin = util, kb
    for name, mod in {"pynput": pkg, "pynput._util": pkg._util, "pynput._util.darwin": util,
                      "pynput.keyboard": pkg.keyboard, "pynput.keyboard._darwin": kb}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, "platform", "darwin")

    assert macapp.harden_pynput() is True
    assert calls == [1]                                   # looked up once, here
    for ctx in (util.keycode_context, kb.keycode_context):
        with ctx() as value:
            assert value == ("kbd-type", b"layout-bytes")
    assert calls == [1]                                   # never again (e.g. on a listener thread)


# ------------------------------------------------------------------ version
def test_app_version_from_bundle_plist(monkeypatch, tmp_path):
    assert paths.app_version() == "source"
    contents = tmp_path / "Minecraft Biogaming.app" / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    exe = contents / "MacOS" / "Minecraft Biogaming"
    exe.write_text("")
    with open(contents / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleShortVersionString": "0.3.0"}, fh)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert paths.app_version() == "0.3.0"
