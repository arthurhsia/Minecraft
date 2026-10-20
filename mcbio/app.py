"""Main loop and command-line interface: ``python -m mcbio [options]``.

Pipeline per frame (the instrumentation chain)::

    source.grab()  ->  Tracker.process()  ->  features.flatten(.., calib)  ->  Mapper.update()
    (camera sensor)    (landmarks+filter)     (calibrated signals)           (thresholds/hysteresis)
                   ->  pause gate  ->  backend.apply()  ->  HUD / logger / protocol
                                       (HID output)

Safety rules implemented here:

* The app starts **ARMING**: a 3-second "STARTING IN 3" countdown during which nothing is
  sent, so the user can switch to the Minecraft window. Every resume is armed the same way.
* **F8** (global hotkey via pynput, if permitted), **P/Space** in the HUD window, or
  **both palms open** (profile ``pause`` trigger) toggle pause. Paused = all released.
* **Esc**/**Q** in the HUD window, closing the window, or **Ctrl-C** quits.
* ``backend.release_all()`` and ``backend.close()`` always run in ``finally``.

Latency is measured per frame from the moment the source returned the frame
(``time.perf_counter()`` right after capture) until ``backend.apply()`` returned.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import signal
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol as TypingProtocol

import numpy as np

from mcbio.types import Calibration, Commands, Features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = ROOT / "profiles" / "face_hand.yaml"

LIVE, PAUSED, ARMING, CALIBRATING = "LIVE", "PAUSED", "ARMING", "CALIBRATING"


# ======================================================================================
# Frame sources
# ======================================================================================
class FrameSource(TypingProtocol):
    """Anything that yields (frame or None, t, Features or None). Features None = end."""
    mirrored: bool

    def grab(self) -> tuple[np.ndarray | None, float, Features | None]: ...

    def close(self) -> None: ...


class CameraSource:
    """Webcam + mediapipe trackers (Agent A's ``tracking.Camera`` and ``tracking.Tracker``)."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 use_hands: bool = True, use_face: bool = True) -> None:
        from mcbio import tracking   # lazy: needs opencv + mediapipe + model files

        self.camera = tracking.Camera(index, width=width, height=height, mirror=True)
        try:
            self.tracker = tracking.Tracker(use_hands=use_hands, use_face=use_face,
                                            frames_mirrored=True)
        except Exception:
            self.camera.close()
            raise
        self.mirrored = True          # Camera(mirror=True) already flips the image
        self._fails = 0

    def grab(self) -> tuple[np.ndarray | None, float, Features | None]:
        ok, frame, t = self.camera.read()
        if not ok or frame is None:
            self._fails += 1
            if self._fails > 30:
                raise RuntimeError("camera stopped delivering frames (try --camera 1)")
            return None, t, Features(t=t)          # empty features -> everything released
        self._fails = 0
        return frame, t, self.tracker.process(frame, t)

    def close(self) -> None:
        for obj in (getattr(self, "tracker", None), getattr(self, "camera", None)):
            if obj is not None:
                try:
                    obj.close()
                except Exception:   # noqa: BLE001 - closing must not raise
                    pass


class SyntheticSource:
    """Wraps :class:`mcbio.synthetic.SyntheticFeatures`; simulated time, optional realtime pacing."""

    def __init__(self, seed: int = 0, mode: str = "script", fps: float = 30.0,
                 realtime: bool = False, size: tuple[int, int] = (1280, 720)) -> None:
        from mcbio.hud import blank_frame
        from mcbio.synthetic import SyntheticFeatures

        self.synth = SyntheticFeatures(seed=seed, fps=fps, mode=mode)
        self.realtime = realtime
        self.mirrored = True
        self._bg = blank_frame(*size)
        self._wall0: float | None = None

    def grab(self) -> tuple[np.ndarray | None, float, Features | None]:
        f = self.synth.next()
        if self.realtime:
            if self._wall0 is None:
                self._wall0 = time.perf_counter() - f.t
            delay = self._wall0 + f.t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        return self._bg, f.t, f

    def cue(self, label: str, t: float) -> None:
        self.synth.cue(label, t)

    def calibration(self) -> Calibration:
        return self.synth.calibration()

    def close(self) -> None:
        pass


# ======================================================================================
# Global hotkey (F8)
# ======================================================================================
class GlobalHotkey:
    """System-wide hotkey using a pynput keyboard Listener (background thread).

    macOS: the terminal/IDE needs *Input Monitoring* permission, otherwise the listener
    silently receives nothing. If pynput is unavailable (e.g. no display) we degrade to the
    HUD-window keys (P / Space) and report ``ok = False``.
    """

    def __init__(self, key: str = "f8") -> None:
        self.key = key
        self.ok = False
        self.error: str | None = None
        self._pressed = threading.Event()
        self._listener: Any = None

    def start(self) -> bool:
        try:
            from pynput import keyboard   # lazy: needs a display / permissions
            target = getattr(keyboard.Key, self.key)

            def on_press(k: Any) -> None:
                if k == target:
                    self._pressed.set()

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
            self.ok = True
        except Exception as e:   # noqa: BLE001 - ImportError, display errors, ...
            self.error = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            self.ok = False
        return self.ok

    def poll(self) -> bool:
        """True once per key press since the last poll."""
        if self._pressed.is_set():
            self._pressed.clear()
            return True
        return False

    def press(self) -> None:
        """Simulate a press (tests / HUD key)."""
        self._pressed.set()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:   # noqa: BLE001
                pass
            self._listener = None


# ======================================================================================
# Calibration flow (on-screen, non-blocking state machine)
# ======================================================================================
class CalibrationFlow:
    """Two-step calibration driven by the main loop (so the HUD keeps updating).

    1. "Sit relaxed, look at the screen, rest your control hand where it's comfortable"
       3-2-1 countdown, then ~2 s neutral capture (Agent B's ``calibration.Calibrator``).
    2. "Move your hand to a comfortable edge" countdown, then ~2.5 s range capture
       (``calibration.capture_range`` -> palm_range = 90th percentile reach).
    """

    def __init__(self, dominant_hand: str = "Right", countdown_s: float = 3.0,
                 neutral_s: float = 2.0, range_s: float = 2.5,
                 require_hand: bool = False, max_retries: int = 3) -> None:
        from mcbio import calibration as cal_mod

        # If the profile steers with the hand, the joystick centre MUST come from
        # where the hand actually rests (e.g. beside the face, not in front of it).
        # Never silently fall back to the middle of the frame.
        self.require_hand = require_hand
        self.max_retries = max_retries
        self.retries = 0
        self._neutral_s = neutral_s

        self._cal = cal_mod
        self.dominant_hand = dominant_hand
        self.countdown_s = countdown_s
        self.range_s = range_s
        self.calibrator = cal_mod.Calibrator(duration_s=neutral_s, dominant_hand=dominant_hand)
        self.phase = "ready"
        self._t_phase: float | None = None
        self._range: list[Features] = []
        self.result: Calibration | None = None
        self.error: str | None = None
        self.prompt: tuple[str, str] = ("", "")
        self.progress: float | None = None
        self.count: int | None = None

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def update(self, feats: Features, t: float) -> None:
        if self._t_phase is None:
            self._t_phase = t
        el = t - self._t_phase
        self.progress, self.count = None, None
        if self.phase == "ready":
            if self.retries:
                self.prompt = ("Couldn't see your hand - raise it into view",
                               "Rest it BESIDE your face (not in front) where the camera sees it - retry in")
            else:
                self.prompt = ("Look at the screen, hand up beside your face",
                               "Rest your control hand off to the side, NOT covering your face - capture in")
            self.count = max(1, math.ceil(self.countdown_s - el))
            if el >= self.countdown_s:
                self._next("neutral", t)
        elif self.phase == "neutral":
            self.prompt = ("Hold still...", "capturing your neutral pose")
            finished = self.calibrator.add(feats)
            self.progress = self.calibrator.progress
            if finished and self.require_hand and self._hand_fraction() < 0.5 \
                    and self.retries < self.max_retries:
                self.retries += 1
                self.calibrator = self._cal.Calibrator(duration_s=self._neutral_s,
                                                       dominant_hand=self.dominant_hand)
                self._next("ready", t)
                return
            if finished:
                if self.require_hand and self._hand_fraction() < 0.5:
                    self.error = "hand not seen - joystick centre is a guess, press C to redo"
                try:
                    self.result = self.calibrator.result()
                except Exception as e:   # noqa: BLE001 - CalibrationError: nothing tracked
                    self.error = str(e)
                    self.result = Calibration(dominant_hand=self.dominant_hand)
                self._next("range_ready", t)
        elif self.phase == "range_ready":
            self.prompt = ("Now move your hand to a comfortable edge",
                           "reach as far as feels easy (up, down or sideways) and hold - in")
            self.count = max(1, math.ceil(self.countdown_s - el))
            if el >= self.countdown_s:
                self._next("range", t)
        elif self.phase == "range":
            self.prompt = ("Hold it at the edge...", "measuring your comfortable reach")
            self._range.append(feats)
            self.progress = min(1.0, el / self.range_s)
            if el >= self.range_s:
                assert self.result is not None
                self.result = self._cal.capture_range(self._range, self.result)
                self._next("done", t)
        if self.phase == "done":
            self.prompt = ("Calibration done", self.error or "")

    def _next(self, phase: str, t: float) -> None:
        self.phase = phase
        self._t_phase = t

    def _hand_fraction(self) -> float:
        smp = self.calibrator.samples
        if not smp:
            return 0.0
        return sum(1 for f in smp if any(h.present for h in f.hands)) / len(smp)


# ======================================================================================
# Config + run statistics
# ======================================================================================
@dataclass
class AppConfig:
    """Everything the CLI can set (see :func:`build_parser`)."""
    profile: str = str(DEFAULT_PROFILE)
    backend: str = "pynput"                 # pynput | leonardo | dry
    port: str | None = None                 # serial port or "auto"
    camera: int = 0
    width: int = 1280
    height: int = 720
    calibrate: bool = False
    calib_path: str | None = None
    hand: str | None = None                 # override dominant hand
    hud: bool = True
    synthetic: bool = False
    seed: int = 0
    frames: int | None = None
    log_dir: str | None = None
    protocol: bool = False
    protocol_trials: int = 30
    subject: str = "anon"
    start_delay: float = 3.0
    target_app: str | None = "minecraft"   # only send input while this app is in front
    hotkey: bool | None = None              # None = auto (off for synthetic+dry)
    realtime: bool | None = None            # synthetic pacing; None = auto (on with HUD)
    verbose: bool = True                    # print action edges in headless mode
    quiet: bool = False


@dataclass
class RunStats:
    """Accumulated statistics printed at the end (and stored in session.json)."""
    frames: int = 0
    t_first: float | None = None
    t_last: float | None = None
    wall_start: float = field(default_factory=time.perf_counter)
    latencies_ms: list[float] = field(default_factory=list)
    onsets: Counter = field(default_factory=Counter)
    held_frames: Counter = field(default_factory=Counter)
    taps: Counter = field(default_factory=Counter)
    pause_toggles: int = 0
    live_frames: int = 0
    cues_done: int = 0
    cues_correct: int = 0
    released_cleanly: bool = False

    def summary(self) -> dict[str, Any]:
        sim = (self.t_last - self.t_first) if (self.t_first is not None and self.t_last is not None) else 0.0
        wall = time.perf_counter() - self.wall_start
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.zeros(1)
        dt = sim / max(1, self.frames - 1)
        return {
            "frames": self.frames,
            "duration_s": round(sim, 3),
            "wall_s": round(wall, 3),
            "fps": round(1.0 / dt, 1) if dt > 0 else 0.0,
            "processing_fps": round(self.frames / wall, 1) if wall > 0 else 0.0,
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
            "latency_ms_max": round(float(lat.max()), 3),
            "live_frames": self.live_frames,
            "pause_toggles": self.pause_toggles,
            "actions": {a: {"onsets": n, "held_s": round(self.held_frames[a] * dt, 2)}
                        for a, n in sorted(self.onsets.items())},
            "taps": dict(self.taps),
            "cues_done": self.cues_done,
            "cues_correct": self.cues_correct,
            "released_cleanly": self.released_cleanly,
        }


# ======================================================================================
# The application
# ======================================================================================
class App:
    """Wires source -> flatten -> mapper -> pause gate -> backend (+ HUD, logger, protocol).

    All collaborators can be injected (tests); otherwise they are built from ``config``.
    """

    def __init__(self, config: AppConfig, *, source: Any = None, backend: Any = None,
                 mapper: Any = None, calibration: Calibration | None = None,
                 hud_window: Any = None, print_fn: Callable[[str], None] = print) -> None:
        self.cfg = config
        self.print = (lambda *_a, **_k: None) if config.quiet else print_fn
        self._source = source
        self._backend = backend
        self._mapper = mapper
        self.calib = calibration
        self._hud_window = hud_window
        self.state = ARMING
        self.arm_until = 0.0
        self.stats = RunStats()
        self.stop = False
        self.flow: CalibrationFlow | None = None
        self.logger: Any = None
        self.protocol: Any = None
        self.hotkey: GlobalHotkey | None = None
        self.profile: Any = None
        self.message: tuple[str, float] | None = None
        self._prev_mapper_paused = False
        self._prev_actions: set[str] = set()
        self.last_cmds: Commands = Commands.idle(paused=True)

    # ------------------------------------------------------------------ setup
    def _setup(self) -> None:
        from mcbio import features as feat_mod
        from mcbio.mapping import Mapper, Profile

        cfg = self.cfg
        self._flatten = feat_mod.flatten
        if self._mapper is None:
            self.profile = Profile.load(cfg.profile)
            self._mapper = Mapper(self.profile)
        else:
            self.profile = getattr(self._mapper, "profile", None)
        self.profile_name = getattr(self.profile, "name", Path(cfg.profile).stem)
        self.hud_debug = self._hud_thresholds()
        psrc = getattr(getattr(self.profile, "pause", None), "source", "")
        self.pause_hint = {"gesture.both_open": "show both open palms", "gesture.smile": "smile (hold)",
                           }.get(psrc, f"hold {psrc}" if psrc else "press P")

        if self._source is None:
            if cfg.synthetic:
                realtime = cfg.realtime if cfg.realtime is not None else cfg.hud
                self._source = SyntheticSource(seed=cfg.seed, realtime=realtime,
                                               mode="responsive" if cfg.protocol else "script",
                                               size=(cfg.width, cfg.height))
            else:
                self._source = CameraSource(cfg.camera, cfg.width, cfg.height)
        self._load_or_start_calibration()

        if self._backend is None:
            self._backend = make_backend(cfg.backend, cfg.port,
                                         getattr(self.profile, "bindings_keys", None),
                                         target_app=cfg.target_app)
        if cfg.hud and self._hud_window is None:
            from mcbio.hud import HudWindow
            self._hud_window = HudWindow("mcbio - Minecraft face+hand controller")
            if not self._hud_window.available:
                self.print(f"[hud] no window available ({getattr(self._hud_window, 'error', '')}); running headless")
                cfg.hud = False          # print action edges / toasts to the console instead
        want_hotkey = cfg.hotkey if cfg.hotkey is not None else not (cfg.synthetic and cfg.backend == "dry")
        if want_hotkey:
            self.hotkey = GlobalHotkey("f8")
            if not self.hotkey.start():
                self.print(f"[hotkey] global F8 unavailable ({self.hotkey.error}); "
                           "use P/Space in the HUD window. macOS: grant Input Monitoring.")
        if cfg.log_dir or cfg.protocol:
            from mcbio.logger import SessionLogger
            self.logger = SessionLogger(cfg.log_dir or "sessions", cfg.subject, meta={
                "profile": self.profile_name, "profile_path": str(cfg.profile),
                "backend": cfg.backend, "synthetic": cfg.synthetic, "protocol": cfg.protocol,
                "argv": sys.argv, "calibration": dataclasses.asdict(self.calib) if self.calib else None,
            })
            self.print(f"[log] writing session to {self.logger.path}")
        if cfg.protocol:
            from mcbio.logger import Protocol
            cue_cb = getattr(self._source, "cue", None)
            self.protocol = Protocol(seed=f"{cfg.subject}-{cfg.seed}", n_trials=cfg.protocol_trials,
                                     on_cue=cue_cb)

    def _calib_path(self) -> Path:
        from mcbio import calibration as cal_mod
        if self.cfg.calib_path:
            return Path(self.cfg.calib_path).expanduser()
        base = Path(cal_mod.DEFAULT_PATH)
        if self.cfg.subject and self.cfg.subject != "anon":
            return base.with_name(f"calibration_{self.cfg.subject}.json")
        return base

    def _load_or_start_calibration(self) -> None:
        from mcbio import calibration as cal_mod
        cfg = self.cfg
        if self.calib is None and not cfg.calibrate:
            if cfg.synthetic and hasattr(self._source, "calibration"):
                self.calib = self._source.calibration()
            else:
                try:
                    self.calib = cal_mod.load(self._calib_path())
                    self.print(f"[calib] loaded {self._calib_path()}")
                except FileNotFoundError:
                    self.print("[calib] no saved calibration - starting calibration")
                except Exception as e:   # noqa: BLE001 - corrupt file
                    self.print(f"[calib] could not read {self._calib_path()} ({e}) - recalibrating")
        if self.calib is not None and cfg.hand:
            self.calib = dataclasses.replace(self.calib, dominant_hand=cfg.hand)
        if self.calib is None:
            self.start_calibration()

    def start_calibration(self) -> None:
        hand = self.cfg.hand or (self.calib.dominant_hand if self.calib else "Right")
        uses_hand = any(str(getattr(b, "source", "")).startswith(("hand.", "gesture.fist", "gesture.pinch"))
                        for b in getattr(self.profile, "bindings", []) or [])
        self.flow = CalibrationFlow(dominant_hand=hand, require_hand=uses_hand)
        self._enter(CALIBRATING, None)

    def _hud_thresholds(self) -> dict[str, Any]:
        """Pull deadzones/thresholds from the profile so the HUD gauges match reality."""
        dbg: dict[str, Any] = {}
        prof = self.profile
        if prof is None:
            return dbg
        th: dict[str, float] = {}
        for b in getattr(prof, "bindings", []):
            if b.source == "hand.palm_dy" and b.action == "forward":
                dbg["move_deadzone"], dbg["move_deadzone_out"] = abs(b.on), abs(b.off)
            if b.source == "face.blend.jawOpen":
                th["jaw_on"], th["jaw_off"] = b.on, b.off
            if b.source == "face.blend.browInnerUp":
                th["brow_on"], th["brow_off"] = b.on, b.off
        g = getattr(prof, "gestures", {}) or {}
        if "fist" in g:
            th["fist_on"], th["fist_off"] = g["fist"].get("theta_on", 0.6), g["fist"].get("theta_off", 0.4)
        if "pinch" in g:
            th["pinch_on"], th["pinch_off"] = g["pinch"].get("on", 0.3), g["pinch"].get("off", 0.45)
        look = getattr(prof, "look", None)
        if look is not None:
            dbg["look_deadzone_deg"] = look.deadzone_deg
        dbg["thresholds"] = th
        return dbg

    # ------------------------------------------------------------------ pause logic
    def _enter(self, state: str, t: float | None) -> None:
        """Switch app state; keeps the Mapper's own pause flag in sync."""
        prev = self.state
        self.state = state
        if state == ARMING and t is not None:
            self.arm_until = t + self.cfg.start_delay
        if state in (PAUSED, CALIBRATING) and self._backend is not None:
            self._backend.release_all()
        if self._mapper is not None and hasattr(self._mapper, "set_paused"):
            want = state in (PAUSED, CALIBRATING)
            if bool(getattr(self._mapper, "paused", False)) != want:
                self._mapper.set_paused(want)
            self._prev_mapper_paused = want
        if prev != state and t is not None and (prev in (LIVE, PAUSED, ARMING)):
            self.stats.pause_toggles += int(state == PAUSED or prev == PAUSED)

    def toggle_pause(self, t: float) -> None:
        """F8 / P / both-palms: PAUSED -> ARMING (countdown) ; LIVE/ARMING -> PAUSED."""
        if self.state == CALIBRATING:
            return
        if self.state == PAUSED:
            self._enter(ARMING, t)
            self._toast("resuming...", t)
        else:
            self._enter(PAUSED, t)
            self._toast("paused - everything released", t)

    def _toast(self, msg: str, t: float) -> None:
        self.message = (msg, t + 2.0)
        if not self.cfg.hud:
            self.print(f"t={t:7.2f}s  [{msg}]")

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict[str, Any]:
        """Run until --frames, end of protocol, Esc, or Ctrl-C. Returns the summary dict."""
        old_handler = None
        try:
            if threading.current_thread() is threading.main_thread():
                old_handler = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        except (ValueError, OSError):
            old_handler = None
        t_last = 0.0
        try:
            self._setup()
            first = True
            while not self.stop:
                frame, t, feats = self._source.grab()
                t_cap = time.perf_counter()
                if feats is None:
                    break                                  # source exhausted
                if first:
                    if self.state == ARMING:
                        self.arm_until = t + self.cfg.start_delay
                    first = False
                t_last = t
                out, flat = self._step(feats, t)
                self._backend.apply(out)
                latency_ms = (time.perf_counter() - t_cap) * 1000.0
                self.last_cmds = out
                self._after_frame(frame, feats, out, flat, t, latency_ms)
                if self.cfg.frames is not None and self.stats.frames >= self.cfg.frames:
                    break
                if self.protocol is not None and self.protocol.finished:
                    self.print("[protocol] all cues done")
                    break
        except KeyboardInterrupt:
            self.print("\n[ctrl-c] stopping")
        finally:
            self._shutdown(t_last)
            if old_handler is not None:
                try:
                    signal.signal(signal.SIGTERM, old_handler)
                except (ValueError, OSError):
                    pass
        return self.summary

    def _step(self, feats: Features, t: float) -> tuple[Commands, dict[str, float]]:
        """One frame of processing; returns (commands actually sent, flat features)."""
        # --- calibration mode: capture, nothing is sent
        if self.state == CALIBRATING and self.flow is not None:
            self.flow.update(feats, t)
            if self.flow.done:
                self.calib = self.flow.result
                if not self.cfg.synthetic:
                    try:
                        from mcbio import calibration as cal_mod
                        p = cal_mod.save(self.calib, self._calib_path())
                        self.print(f"[calib] saved {p}")
                    except Exception as e:   # noqa: BLE001
                        self.print(f"[calib] could not save: {e}")
                if self.logger is not None and self.calib is not None:
                    self.logger.meta["calibration"] = dataclasses.asdict(self.calib)
                self.flow = None
                self._enter(ARMING, t)
            calib = self.calib or Calibration()
            flat = self._flatten(feats, calib)
            idle = Commands.idle(paused=True)
            idle.debug = {**self.hud_debug}
            return idle, flat

        assert self.calib is not None
        flat = self._flatten(feats, self.calib)
        cmds = self._mapper.update(flat, t)
        cmds.debug = {**self.hud_debug, **(cmds.debug or {})}

        # --- external pause toggles: F8 hotkey (global) ...
        if self.hotkey is not None and self.hotkey.poll():
            self.toggle_pause(t)
        # ... and the profile's pause gesture (Mapper flips its own `paused` flag)
        mp = bool(cmds.paused)
        if mp != self._prev_mapper_paused:
            self._prev_mapper_paused = mp
            if mp and self.state in (LIVE, ARMING):
                self._enter(PAUSED, t)
                self._toast("paused (both palms)", t)
            elif not mp and self.state == PAUSED:
                self._enter(ARMING, t)
                self._toast("resuming (both palms)", t)
        if self.state == ARMING and t >= self.arm_until:
            self.state = LIVE
            self._toast("LIVE", t)

        if self.state != LIVE:
            gated = Commands.idle(paused=True)
            gated.debug = cmds.debug
            return gated, flat
        return cmds, flat

    def _after_frame(self, frame: np.ndarray | None, feats: Features, out: Commands,
                     flat: dict[str, float], t: float, latency_ms: float) -> None:
        from mcbio.logger import active_actions
        st = self.stats
        if st.t_first is None:
            st.t_first = t
        prev_t = st.t_last
        st.t_last = t
        st.frames += 1
        st.latencies_ms.append(latency_ms)
        live = self.state == LIVE
        st.live_frames += int(live)
        fps = 0.0
        if prev_t is not None and t > prev_t:
            inst = 1.0 / (t - prev_t)
            self._fps = inst if not hasattr(self, "_fps") else 0.9 * self._fps + 0.1 * inst
            fps = self._fps
        actions = active_actions(out)
        for a in actions - self._prev_actions:
            st.onsets[a] += 1
            if not self.cfg.hud and self.cfg.verbose:
                self.print(f"t={t:7.2f}s  +{a}")
        if not self.cfg.hud and self.cfg.verbose:
            for a in sorted(self._prev_actions - actions):
                self.print(f"t={t:7.2f}s  -{a}")
        for a in actions:
            st.held_frames[a] += 1
        for tap in out.taps:
            st.taps[tap] += 1
        self._prev_actions = actions

        cue = None
        if self.protocol is not None:
            for row in self.protocol.update(t, actions, live=live):
                st.cues_done += 1
                st.cues_correct += int(bool(row["correct"]))
                if self.logger is not None:
                    self.logger.log_cue(row)
                if not self.cfg.hud and self.cfg.verbose:
                    rt = f"{row['rt_ms']:.0f} ms" if row["rt_ms"] is not None else "-"
                    self.print(f"t={t:7.2f}s  cue {row['cue']:<10} -> {row['outcome']:<6} {rt}")
            cue = self.protocol.current
        if self.logger is not None:
            self.logger.log_frame(t, latency_ms, fps, not live, out, flat, cue)
        if self._hud_window is not None and getattr(self._hud_window, "available", False):
            self._draw_and_show(frame, feats, out, flat, t, fps, latency_ms, actions)

    def _draw_and_show(self, frame: np.ndarray | None, feats: Features, out: Commands,
                       flat: dict[str, float], t: float, fps: float, latency_ms: float,
                       actions: set[str]) -> None:
        from mcbio.hud import HudState, blank_frame, draw_hud
        img = frame if frame is not None else blank_frame(self.cfg.width, self.cfg.height, "NO CAMERA")
        msg = None
        if self.message and t < self.message[1]:
            msg = self.message[0]
        st = HudState(
            status=self.state, fps=fps, latency_ms=latency_ms,
            countdown=max(0.0, self.arm_until - t) if self.state == ARMING else None,
            profile=self.profile_name, backend=self.cfg.backend + (" (synthetic)" if self.cfg.synthetic else ""),
            message=msg, hotkey_ok=bool(self.hotkey and self.hotkey.ok), actions=actions,
            pause_hint=self.pause_hint,
            neutral_palm=self.calib.neutral_palm if self.calib else None,
            palm_range=self.calib.palm_range if self.calib else None,
        )
        if self.flow is not None:
            st.prompt, st.prompt_progress, st.prompt_count = self.flow.prompt, self.flow.progress, self.flow.count
        if self.protocol is not None and self.protocol.current:
            st.cue = self.protocol.current
            st.cue_progress = self.protocol.progress(t)
            st.cue_info = f"trial {self.protocol.completed + 1}/{self.protocol.n_trials}"
        img = draw_hud(img, feats, out, out.debug, flat, st,
                       mirror=not getattr(self._source, "mirrored", True))
        key = self._hud_window.show(img)
        if key in (27, ord("q")):                  # Esc / q
            self.stop = True
        elif key in (ord("p"), ord(" ")):
            self.toggle_pause(t)
        elif key == ord("c") and self.state != CALIBRATING:
            self.start_calibration()

    # ------------------------------------------------------------------ shutdown
    def _shutdown(self, t: float) -> None:
        """Release everything no matter what, then close resources (each step guarded)."""
        if self._backend is not None:
            for step in ("release_all", "close"):
                try:
                    getattr(self._backend, step)()
                except Exception as e:   # noqa: BLE001
                    print(f"[shutdown] backend.{step} failed: {e}", file=sys.stderr)
            self.stats.released_cleanly = True
        for obj, name in ((self.hotkey, "stop"), (self._hud_window, "close"), (self._source, "close")):
            if obj is not None:
                try:
                    getattr(obj, name)()
                except Exception:   # noqa: BLE001
                    pass
        self.summary = self.stats.summary()
        if self.logger is not None:
            try:
                self.logger.close(self.summary, t=t)
            except Exception as e:   # noqa: BLE001
                print(f"[shutdown] logger close failed: {e}", file=sys.stderr)


def _raise_keyboard_interrupt(*_: Any) -> None:
    raise KeyboardInterrupt


# ======================================================================================
# Backends + CLI
# ======================================================================================
def make_backend(name: str, port: str | None = None, bindings_keys: dict[str, str] | None = None,
                 target_app: str | None = "minecraft") -> Any:
    """Construct an output backend; real backends are focus-gated to ``target_app``.

    With ``target_app`` set (default "minecraft"), input is only sent while an app
    whose name contains it is frontmost; switching away releases everything.
    """
    from mcbio import outputs
    if name == "dry":
        return outputs.DryRunOutput(bindings_keys)

    def gate(backend: Any) -> Any:
        if not target_app or outputs.frontmost_app_name() is None:   # non-macOS: cannot tell
            return backend
        def say(focused: bool, app: str | None) -> None:
            print(("[output ON]  " if focused else "[output OFF] ") +
                  (f"{app} is in front" if focused else f"{app or 'another app'} is in front - "
                   f"waiting for {target_app.title()} (nothing is sent to other apps)"), flush=True)
        return outputs.FocusGatedOutput(backend, targets=(target_app,), on_change=say)

    if name == "pynput":
        return gate(outputs.PynputOutput(bindings_keys))
    if name == "leonardo":
        p = None if port in (None, "auto") else port
        if p is None:
            p = outputs.find_leonardo_port()
            if p is None:
                raise SystemExit("No Arduino Leonardo found. Plug it in, flash arduino/leonardo_hid, "
                                 "or pass --port /dev/cu.usbmodemXXXX")
        return gate(outputs.SerialLeonardoOutput(p, bindings_keys=bindings_keys))
    raise SystemExit(f"unknown backend {name!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mcbio",
        description="Control Minecraft Education Edition with your face and hand (webcam).",
        epilog="Keys: F8 (global) or P/Space in the HUD = pause/resume, C = recalibrate, Esc/Q = quit.")
    p.add_argument("--profile", default=str(DEFAULT_PROFILE), help="control profile YAML (default: face_hand)")
    p.add_argument("--backend", choices=("pynput", "leonardo", "dry"), default="pynput",
                   help="output: pynput (this Mac), leonardo (Arduino USB HID), dry (print only)")
    p.add_argument("--port", default=None, help="serial port for --backend leonardo, or 'auto'")
    p.add_argument("--camera", type=int, default=0, help="camera index (try 1 if 0 is the wrong camera)")
    p.add_argument("--size", default="1280x720", help="camera resolution WxH")
    p.add_argument("--calibrate", action="store_true", help="run calibration even if one is saved")
    p.add_argument("--calib", dest="calib_path", default=None, help="calibration JSON path")
    p.add_argument("--hand", choices=("Left", "Right"), default=None, help="dominant (control) hand")
    p.add_argument("--no-hud", action="store_true", help="no window (headless)")
    p.add_argument("--synthetic", action="store_true", help="use the synthetic user (no camera/mediapipe)")
    p.add_argument("--seed", type=int, default=0, help="synthetic RNG seed")
    p.add_argument("--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--log-dir", default=None, help="write session logs here (e.g. sessions/)")
    p.add_argument("--protocol", action="store_true", help="run the cue/reaction-time experiment")
    p.add_argument("--trials", type=int, default=30, help="number of protocol cues")
    p.add_argument("--subject", default="anon", help="participant ID for logs/calibration")
    p.add_argument("--start-delay", type=float, default=3.0, help="arming countdown seconds")
    p.add_argument("--target-app", default="minecraft",
                   help="only send input while the frontmost app's name contains this (default: minecraft)")
    p.add_argument("--any-app", action="store_true", help="send input to whatever app is in front (not recommended)")
    p.add_argument("--dry-run", action="store_true", help="same as --backend dry")
    p.add_argument("--no-hotkey", action="store_true", help="disable the global F8 listener")
    p.add_argument("--realtime", action="store_true", help="pace synthetic frames at 30 fps")
    p.add_argument("--quiet", action="store_true", help="print only the summary")
    return p


def config_from_args(argv: list[str] | None = None) -> AppConfig:
    a = build_parser().parse_args(argv)
    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size must look like 1280x720, got {a.size!r}")
    return AppConfig(
        profile=a.profile, backend="dry" if a.dry_run else a.backend, port=a.port,
        camera=a.camera, width=w, height=h, calibrate=a.calibrate, calib_path=a.calib_path,
        hand=a.hand, hud=not a.no_hud, synthetic=a.synthetic, seed=a.seed, frames=a.frames,
        log_dir=a.log_dir, protocol=a.protocol, protocol_trials=a.trials, subject=a.subject,
        start_delay=a.start_delay, target_app=None if a.any_app else a.target_app, hotkey=False if a.no_hotkey else None,
        realtime=True if a.realtime else None, quiet=a.quiet,
    )


def format_summary(s: dict[str, Any]) -> str:
    """Human-readable end-of-run summary."""
    acts = ", ".join(f"{a} x{v['onsets']} ({v['held_s']:.1f}s)" for a, v in s["actions"].items()) or "none"
    lines = [
        "---- mcbio summary ----",
        f"frames {s['frames']}  ({s['duration_s']:.1f} s stream, {s['fps']:.1f} fps; "
        f"processing {s['processing_fps']:.0f} fps)",
        f"latency capture->output  p50 {s['latency_ms_p50']:.2f} ms  p95 {s['latency_ms_p95']:.2f} ms  "
        f"max {s['latency_ms_max']:.2f} ms",
        f"live frames {s['live_frames']}   pause toggles {s['pause_toggles']}",
        f"actions seen: {acts}",
    ]
    if s["taps"]:
        lines.append("taps: " + ", ".join(f"{k} x{v}" for k, v in s["taps"].items()))
    if s["cues_done"]:
        lines.append(f"protocol: {s['cues_correct']}/{s['cues_done']} cues correct")
    lines.append(f"all keys/buttons released: {'yes' if s['released_cleanly'] else 'NO'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(argv)
    app = App(cfg)
    try:
        summary = app.run()
    except FileNotFoundError as e:          # e.g. missing model files / profile
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:               # camera / serial problems
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_summary(summary))
    return 0
