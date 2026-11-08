"""Start screen for the packaged Mac app (also runnable from source: ``python -m mcbio.launcher``).

Double-clicking "Minecraft Biogaming.app" lands here instead of in a Terminal. One
OpenCV window (the same one the HUD then uses) shows:

* a live camera preview with a camera picker (cycle with the arrows or Tab), retrying
  until macOS camera permission is granted;
* the control profile to play with (face + hand, face only, hand only);
* the three macOS permissions, each with a button that opens the right Settings pane;
* a big START button. Calibration and the 3-second countdown follow as usual.

When the controller quits (Esc in its window) the start screen comes back; Esc here
quits the app. Session logs go to ``~/Documents/Minecraft Biogaming/sessions`` and
everything printed goes to ``mcbio.log`` in the same folder (an app has no Terminal).

Command-line arguments bypass the start screen, so the bundled executable still works
as ``python -m mcbio`` does; ``--self-test`` loads the models and exits (used by CI to
check the packaged app).
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mcbio import macapp, macperms, paths

WINDOW = "mcbio - Minecraft face+hand controller"     # shared with the HUD window
W, H = 1280, 720

#: (label, profile file, one-line description)
PROFILES: tuple[tuple[str, str, str], ...] = (
    ("Face + hand", "face_hand.yaml", "Head looks, palm walks, fist mines"),
    ("Face only", "face_only.yaml", "No hands needed"),
    ("Hand only", "hand_only.yaml", "Two hands, no face gestures"),
)
PERMS: tuple[tuple[str, str, str], ...] = (
    ("camera", "Camera", "to see you"),
    ("accessibility", "Accessibility", "to press keys and move the mouse"),
    ("input_monitoring", "Input Monitoring", "for the F8 pause key (optional)"),
)
MAX_CAMERAS = 4
#: (label, --game value) for the Game buttons.
GAMES: tuple[tuple[str, str], ...] = (("Education / Bedrock", "education"), ("Java Edition", "java"))


@dataclass
class StartState:
    profile: int = 0
    game: int = 0
    camera: int = 0
    camera_ok: bool = False
    perms: dict[str, bool | None] = field(default_factory=dict)
    message: str = ""
    message_is_error: bool = False


# ====================================================================== drawing (pure)
def draw_start_screen(preview: np.ndarray | None, st: StartState) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int, int]]]]:
    """Render the start screen; returns (image, clickable regions as (name, (x0, y0, x1, y1)))."""
    import cv2

    from mcbio.hud import AMBER, BLACK, BLUE, DARK, GREEN, GREY, LIT, RED, WHITE, FONT, FONT_BOLD, blank_frame, panel, wrap
    from mcbio.hud import text as _text

    def text(*a: Any, **k: Any) -> Any:          # plain background: no outline needed
        return _text(*a, outline=False, **k)

    img = blank_frame(W, H, "")
    buttons: list[tuple[str, tuple[int, int, int, int]]] = []

    def button(name: str, x: int, y: int, w: int, h: int, label: str, on: bool = False,
               color: tuple[int, int, int] = LIT, scale: float = 0.7) -> None:
        if on:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        else:
            panel(img, x, y, w, h, DARK, 0.85)
        cv2.rectangle(img, (x, y), (x + w, y + h), WHITE if on else GREY, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, scale, 2)
        cv2.putText(img, label, (x + (w - tw) // 2, y + (h + th) // 2), FONT_BOLD, scale,
                    BLACK if on else WHITE, 2, cv2.LINE_AA)
        buttons.append((name, (x, y, x + w, y + h)))

    text(img, paths.APP_NAME, (40, 64), 1.3, WHITE, 3, FONT_BOLD)
    text(img, "Play Minecraft with your face and hands", (40, 100), 0.7, GREY, 2)
    text(img, f"Version {paths.app_version()}", (680, 100), 0.5, GREY, 1)

    # --- camera preview (left)
    px, py, pw, ph = 40, 130, 600, 338
    cv2.rectangle(img, (px - 2, py - 2), (px + pw + 2, py + ph + 2), GREY, 2)
    if preview is not None:
        img[py:py + ph, px:px + pw] = cv2.resize(preview, (pw, ph))
    else:
        text(img, "No camera image yet", (px + pw // 2, py + ph // 2 - 10), 0.8, AMBER, 2, center=True)
        text(img, "Allow camera access if macOS asks,", (px + pw // 2, py + ph // 2 + 25), 0.6, GREY, 1, center=True)
        text(img, "or try the next camera", (px + pw // 2, py + ph // 2 + 50), 0.6, GREY, 1, center=True)
    button("cam_prev", px, py + ph + 16, 56, 48, "<")
    text(img, f"Camera {st.camera}", (px + pw // 2, py + ph + 50), 0.8, WHITE, 2, center=True)
    button("cam_next", px + pw - 56, py + ph + 16, 56, 48, ">")
    text(img, "Tip: sit 50-80 cm away, face lit from the front", (px, py + ph + 100), 0.55, GREY, 1)

    # --- profiles (right)
    rx, rw = 680, 560
    text(img, "Controls", (rx, 150), 0.8, WHITE, 2, FONT_BOLD)
    for i, (label, _, desc) in enumerate(PROFILES):
        y = 160 + i * 52
        button(f"profile_{i}", rx, y, 170, 42, label, on=(i == st.profile), color=BLUE)
        text(img, desc, (rx + 185, y + 27), 0.48, WHITE if i == st.profile else GREY, 1)

    # --- game
    text(img, "Game", (rx, 342), 0.8, WHITE, 2, FONT_BOLD)
    for i, (label, _) in enumerate(GAMES):
        button(f"game_{i}", rx + 90 + i * 240, 314, 230, 42, label, on=(i == st.game), color=BLUE, scale=0.6)

    # --- permissions
    text(img, "Permissions", (rx, 400), 0.8, WHITE, 2, FONT_BOLD)
    for i, (key, label, why) in enumerate(PERMS):
        y = 412 + i * 44
        ok = st.camera_ok if key == "camera" else st.perms.get(key)
        col, word = (GREEN, "OK") if ok else ((AMBER, "?") if ok is None else (RED, "needed"))
        cv2.circle(img, (rx + 12, y + 18), 9, col, -1, cv2.LINE_AA)
        text(img, f"{label}", (rx + 32, y + 25), 0.6, WHITE, 2)
        text(img, f"{word} - {why}", (rx + 210, y + 25), 0.5, col if not ok else GREY, 1)
        if not ok and sys.platform == "darwin":
            button(f"perm_{key}", rx + rw - 150, y + 2, 150, 36, "Open Settings", scale=0.55)

    # --- start
    button("start", rx, 556, rw, 64, "START", on=True, color=GREEN, scale=1.2)
    msg, is_err = st.message, st.message_is_error
    if not msg and GAMES[st.game][1] == "java":
        from mcbio.app import JAVA_SPRINT_NOTE
        msg, is_err = JAVA_SPRINT_NOTE, False
    if msg:
        for i, line in enumerate(wrap(msg, rw, 0.55, 1, FONT)[:2]):
            text(img, line, (rx, 646 + i * 22), 0.55, RED if is_err else AMBER, 1)
    text(img, "Keys: 1/2/3 controls - G game - Tab camera - Enter start - Esc or Cmd-Q quit",
         (40, H - 18), 0.5, GREY, 1)
    data = str(paths.data_dir(create=False)).replace(str(paths.Path.home()), "~")
    if len(data) > 48:
        data = "..." + data[-45:]
    text(img, f"Recordings: {data}", (40, H - 42), 0.5, GREY, 1)
    return img, buttons


def hit(buttons: list[tuple[str, tuple[int, int, int, int]]], x: int, y: int) -> str | None:
    """Name of the button under (x, y), if any."""
    for name, (x0, y0, x1, y1) in buttons:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return None


def controller_args(st: StartState) -> list[str]:
    """CLI arguments for the chosen settings (what START runs)."""
    profile = paths.profiles_dir() / PROFILES[st.profile][1]
    return ["--profile", str(profile), "--camera", str(st.camera), "--calibrate",
            "--game", GAMES[st.game][1],
            "--backend", "pynput", "--log-dir", str(paths.data_dir() / "sessions")]


# ====================================================================== runtime
class _Preview:
    """Camera preview that keeps retrying (macOS grants camera access asynchronously)."""

    def __init__(self) -> None:
        import cv2
        self._cv2 = cv2
        self.cap: Any = None
        self.index = -1
        self._next_try = 0.0

    def frame(self, index: int) -> np.ndarray | None:
        cv2 = self._cv2
        if index != self.index:
            self.release()
            self.index = index
            self._next_try = 0.0
        if self.cap is None:
            if time.monotonic() < self._next_try:
                return None
            self._next_try = time.monotonic() + 1.5
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap = cap
        ok, img = self.cap.read()
        if not ok or img is None:
            self.release()
            return None
        return cv2.flip(img, 1)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _refresh_perms(st: StartState) -> None:
    st.perms = {"accessibility": macperms.accessibility(), "input_monitoring": macperms.input_monitoring()}


def run_controller(st: StartState) -> tuple[str, bool]:
    """Run the controller with the chosen settings; returns (message, is_error) for the start screen."""
    from mcbio.app import App, config_from_args, format_summary
    try:
        cfg = config_from_args(controller_args(st))
        summary = App(cfg).run()
        print(format_summary(summary), flush=True)
        return "Session saved. Press START to play again.", False
    except Exception as e:   # noqa: BLE001 - show it instead of crashing the app
        traceback.print_exc()
        first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        return f"Could not start: {first}", True


def start_screen() -> int:
    """Event loop of the start screen (blocking)."""
    import cv2

    st = StartState()
    macapp.harden_pynput()                  # main thread: read the keyboard layout once
    _refresh_perms(st)
    if st.perms.get("accessibility") is False:
        macperms.prompt_accessibility()          # adds the app to the Accessibility list
    preview = _Preview()
    clicks: list[tuple[int, int]] = []
    buttons: list[tuple[str, tuple[int, int, int, int]]] = []
    next_perm_check = 0.0

    def open_window() -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        macapp.install_quit_menu()        # OpenCV's window has no menu bar: add Quit / Cmd-Q
        cv2.setMouseCallback(WINDOW, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONUP else None)

    open_window()
    while True:
        now = time.monotonic()
        if now >= next_perm_check:
            _refresh_perms(st)
            next_perm_check = now + 1.0
        img = preview.frame(st.camera)
        st.camera_ok = img is not None
        screen, buttons = draw_start_screen(img, st)
        cv2.imshow(WINDOW, screen)
        key = cv2.waitKey(30) & 0xFF
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break                                  # window closed with the mouse
        except cv2.error:
            break

        action = None
        if key in (27, ord("q")) or macapp.quit_requested():
            break
        if key in (13, 10, ord(" ")):
            action = "start"
        elif key in (ord("1"), ord("2"), ord("3")):
            action = f"profile_{key - ord('1')}"
        elif key == 9:
            action = "cam_next"
        elif key == ord("g"):
            action = f"game_{(st.game + 1) % len(GAMES)}"
        while clicks:
            action = hit(buttons, *clicks.pop(0)) or action

        if action == "cam_next":
            st.camera = (st.camera + 1) % MAX_CAMERAS
        elif action == "cam_prev":
            st.camera = (st.camera - 1) % MAX_CAMERAS
        elif action and action.startswith("game_"):
            st.game = int(action.split("_")[1])
            st.message = ""
        elif action and action.startswith("profile_"):
            st.profile = int(action.split("_")[1])
        elif action and action.startswith("perm_"):
            which = action[len("perm_"):]
            if which == "accessibility":
                macperms.prompt_accessibility()
            elif which == "input_monitoring":
                macperms.prompt_input_monitoring()
            macperms.open_settings(which)
            st.message, st.message_is_error = ("Turn on Minecraft Biogaming in the list, then come back "
                                               "(quit and reopen the app if it still says needed)."), False
        elif action == "start":
            if not st.camera_ok:
                st.message, st.message_is_error = "No camera image - allow camera access or pick another camera.", True
                continue
            if st.perms.get("accessibility") is False:
                st.message, st.message_is_error = ("Accessibility is off: the game will not receive any keys. "
                                                   "Click Open Settings first."), True
                continue
            preview.release()                          # the controller opens the camera itself
            st.message, st.message_is_error = run_controller(st)
            if macapp.quit_requested():
                break
            open_window()                              # the HUD closed the shared window
    preview.release()
    cv2.destroyAllWindows()
    return 0


def self_test() -> int:
    """Load every bundled piece without a camera or window (CI check of the packaged app)."""
    import cv2

    from mcbio import tracking
    from mcbio.mapping import Profile

    print(f"{paths.APP_NAME} self-test | frozen={paths.is_frozen()} | resources={paths.resource_dir()}")
    for _, name, _ in PROFILES:
        Profile.load(paths.profiles_dir() / name)
    tr = tracking.Tracker()
    try:
        feats = tr.process(np.zeros((720, 1280, 3), np.uint8), time.monotonic())
    finally:
        tr.close()
    import mediapipe
    if sys.platform == "darwin":                # (pynput needs a display on Linux)
        import pynput  # noqa: F401 - key output
        import Quartz  # noqa: F401 - relative mouse + focus lock
    # Start/stop the global F8 listener exactly as a session does (it crashed the app on
    # newer macOS when it came from pynput). Without Input Monitoring it reports "off".
    from mcbio.app import GlobalHotkey
    hk = GlobalHotkey("f8")
    hardened = macapp.harden_pynput()
    hk_state = "on" if hk.start() else f"off ({hk.error})"
    time.sleep(0.3)
    hk.stop()
    print(f"F8 hotkey: {hk_state} | pynput layout cache: {'on' if hardened else 'off'} | "
          f"version {paths.app_version()}")
    print(f"OK: mediapipe {mediapipe.__version__}, opencv {cv2.__version__}, "
          f"profiles {len(PROFILES)}, tracker ran (face={feats.face.present}, hands={len(feats.hands)})")
    return 0


def _log_to_file() -> None:
    """An app launched from Finder has no Terminal: send prints and errors to mcbio.log."""
    try:
        log = open(paths.data_dir() / "mcbio.log", "a", buffering=1, encoding="utf-8")
    except OSError:
        return
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    sys.stdout = sys.stderr = log


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = [a for a in argv if not a.startswith("-psn_")]   # old Finder process-serial argument
    if argv == ["--self-test"]:
        return self_test()
    if argv:                                                # CLI mode, same as python -m mcbio
        from mcbio.app import main as cli_main
        return cli_main(argv)
    if paths.is_frozen():
        _log_to_file()
    try:
        return start_screen()
    except Exception:   # noqa: BLE001 - log it; a Finder-launched app would just vanish
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
