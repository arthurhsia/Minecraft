"""Map calibrated feature values to game commands, driven by a YAML profile.

Pipeline per frame::

    flat = features.flatten(Features, Calibration)     # dict[str, float]  (Agent A)
    cmds = Mapper(profile).update(flat, t)             # Commands          (this module)
    backend.apply(cmds)                                # outputs.py

A *profile* (``profiles/*.yaml``) describes everything user-tunable:

``bindings``
    list of ``{action, source, op, on, off, min_hold_ms?, min_release_ms?, mode?}``.
    ``source`` is a flat-dict key (e.g. ``hand.palm_dy``, ``face.blend.jawOpen``) or a
    derived gesture (``gesture.fist``, ``gesture.pinch``, ``gesture.open_palm``,
    ``gesture.both_open``, ``gesture.both_blink``, ``gesture.smile``); gestures are
    0/1 so their ``op/on/off`` default to ``">", 0.5, 0.5``.  Several bindings may
    drive the same action (logical OR).
``look``
    ``{yaw_source, pitch_source, deadzone_deg, gain_px_per_deg_s, expo, expo_ref_deg?,
    pitch_gain_px_per_deg_s?, max_px_per_frame, invert_y, mode?}`` – rate ("joystick")
    control of the camera.
``pause``
    ``{source, op?, on?, hold_s}`` – hold this condition to toggle pause.
``mode_toggle`` (optional)
    ``{source, op?, on?, hold_s, modes: [walk, look], initial?}`` – hold to cycle modes;
    bindings / look tagged with ``mode`` are only live in that mode.
``gestures`` (optional)
    threshold overrides for the derived gestures (see :data:`GESTURE_DEFAULTS`).
``bindings_keys``
    logical action -> physical key name, consumed by ``outputs.py``.

Tap actions (``inventory``, ``hotbar_next``, ``hotbar_prev``) are one-shot: they fire once
when their binding turns on (key press + release, or one mouse-wheel step for the hotbar),
and again every ``repeat_ms`` while it stays on, if set.

Presence gating: a source whose prefix is ``face.``/``hand.``/``offhand.`` (or a gesture
built from that tracker) is treated as *absent* while the matching ``*.present`` flag is
0.  Absent sources reset their detectors and release their action immediately, so a
hand leaving the frame never leaves W held down.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .gestures import CurlStateMachine, HoldTimer, Hysteresis
from .outputs import DEFAULT_KEYMAP
from .types import Commands

#: Actions a binding may drive.  Movement names match ``types.MOVE_KEYS``.
KEY_ACTIONS = ("forward", "back", "left", "right", "jump", "sneak", "sprint")
BUTTON_ACTIONS = ("left_button", "right_button")
#: One-shot actions: a press + release on the rising edge (optionally repeated while held).
TAP_ACTIONS = ("inventory", "hotbar_next", "hotbar_prev")
ACTIONS = KEY_ACTIONS + BUTTON_ACTIONS + TAP_ACTIONS

#: Fingers (besides the index) the thumb can pinch; flat keys ``hand.pinch_<finger>``.
FINGER_PINCHES = ("middle", "ring", "little")

#: Opposing pairs: never both held; the one driven harder wins.
OPPOSING = (("forward", "back"), ("left", "right"))

#: Derived gesture name -> presence flag that gates it.
GESTURE_PRESENCE = {
    "gesture.fist": "hand.present",
    "gesture.pinch": "hand.present",
    "gesture.pinch_middle": "hand.present",
    "gesture.pinch_ring": "hand.present",
    "gesture.pinch_little": "hand.present",
    "gesture.open_palm": "hand.present",
    "gesture.both_open": "offhand.present",   # also needs hand.present, checked inside
    "gesture.both_blink": "face.present",
    "gesture.smile": "face.present",
}

#: Default thresholds for the derived gestures (override under ``gestures:`` in YAML).
GESTURE_DEFAULTS: dict[str, dict[str, float]] = {
    # CurlStateMachine on hand.curl_mean (0 open .. 1 fist); see gestures.py.
    "fist": {"theta_on": 0.6, "theta_off": 0.4, "eps_theta": 0.25,
             "eps_thetadot": 1.0, "tau_s": 0.04, "tau_vel_s": 0.06},
    # hand.pinch < on to start, > off to stop; ignored while the hand is a fist
    # (curl_mean >= max_curl_mean) because a fist also brings thumb and index together.
    "pinch": {"on": 0.30, "off": 0.45, "min_hold_ms": 80, "max_curl_mean": 0.5},
    # Thumb to middle / ring / little fingertip (gesture.pinch_<finger>). Same rule as the
    # index pinch; in addition only the fingertip NEAREST the thumb can count, so touching
    # the middle finger (with the index right beside it) never also fires the index pinch.
    "finger_pinch": {"on": 0.30, "off": 0.45, "min_hold_ms": 100, "max_curl_mean": 0.5},
    # Open palm: curl_mean < on (release > off) and thumb away from index (pinch >= min_pinch).
    "open_palm": {"on": 0.20, "off": 0.30, "min_pinch": 0.5},
    # Both eyes closed: min(eyeBlinkLeft, eyeBlinkRight) > on (release < off).
    "both_blink": {"on": 0.5, "off": 0.3},
    # Both mouth corners smiling: min(mouthSmileLeft, mouthSmileRight) > on.
    "smile": {"on": 0.5, "off": 0.3},
}


class _ProfileLoader(yaml.SafeLoader):
    """SafeLoader where only ``true``/``false`` are booleans.

    YAML 1.1 (PyYAML) reads bare ``on`` / ``off`` / ``yes`` / ``no`` as booleans, which
    would turn the binding keys ``on:`` / ``off:`` into ``True`` / ``False``.
    """


_ProfileLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_ProfileLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)

# Frames further apart than this (e.g. after a stall) are treated as this long, so the
# camera never jumps by a whole second's worth of motion.
MAX_DT_S = 0.1


def presence_key(source: str) -> str | None:
    """Return the ``*.present`` flag that gates ``source`` (None = always present)."""
    if source in GESTURE_PRESENCE:
        return GESTURE_PRESENCE[source]
    for prefix in ("face", "hand", "offhand"):
        if source.startswith(prefix + ".") and source != f"{prefix}.present":
            return f"{prefix}.present"
    return None


def _validate_source(source: Any, where: str) -> str:
    if not isinstance(source, str) or not source:
        raise ValueError(f"{where}: source must be a non-empty string")
    if source.startswith("gesture."):
        if source not in GESTURE_PRESENCE:
            raise ValueError(f"{where}: unknown gesture {source!r}; "
                             f"known: {sorted(GESTURE_PRESENCE)}")
    elif not source.startswith(("face.", "hand.", "offhand.", "hands.")):
        raise ValueError(f"{where}: source {source!r} must start with face./hand./"
                         f"offhand./hands./gesture.")
    return source


@dataclass
class Binding:
    """One threshold rule: ``source`` crossing ``on``/``off`` drives ``action``."""
    action: str
    source: str
    op: str = ">"
    on: float = 0.5
    off: float = 0.5
    min_hold_ms: float = 0.0
    min_release_ms: float = 0.0
    mode: str | None = None
    repeat_ms: float = 0.0          # tap actions only: re-fire every repeat_ms while held (0 = once)

    @classmethod
    def from_dict(cls, d: dict, index: int) -> "Binding":
        where = f"bindings[{index}]"
        action = d.get("action")
        if action not in ACTIONS:
            raise ValueError(f"{where}: action {action!r} not in {ACTIONS}")
        source = _validate_source(d.get("source"), where)
        is_gesture = source.startswith("gesture.")
        if not is_gesture and ("on" not in d or "off" not in d):
            raise ValueError(f"{where}: 'on' and 'off' are required for {source!r}")
        b = cls(
            action=action,
            source=source,
            op=str(d.get("op", ">")),
            on=float(d.get("on", 0.5)),
            off=float(d.get("off", d.get("on", 0.5) if is_gesture else 0.5)),
            min_hold_ms=float(d.get("min_hold_ms", 0)),
            min_release_ms=float(d.get("min_release_ms", 0)),
            mode=d.get("mode"),
            repeat_ms=float(d.get("repeat_ms", 0)),
        )
        if b.repeat_ms < 0:
            raise ValueError(f"{where}: repeat_ms must be >= 0")
        b.make_detector()           # validates op / threshold ordering early
        return b

    def make_detector(self) -> Hysteresis:
        return Hysteresis(self.on, self.off, self.op, self.min_hold_ms, self.min_release_ms)

    def drive(self, value: float) -> float:
        """How hard the binding is pushed past its release threshold (unit-free)."""
        span = max(abs(self.on - self.off), 1e-6)
        return (value - self.off) / span if self.op == ">" else (self.off - value) / span


@dataclass
class HoldTrigger:
    """``pause`` / ``mode_toggle`` condition: ``source op on`` held for ``hold_s``."""
    source: str
    hold_s: float
    op: str = ">"
    on: float = 0.5

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "HoldTrigger":
        op = str(d.get("op", ">"))
        if op not in (">", "<"):
            raise ValueError(f"{where}: op must be '>' or '<'")
        return cls(source=_validate_source(d.get("source"), where),
                   hold_s=float(d.get("hold_s", 1.0)), op=op, on=float(d.get("on", 0.5)))

    def cond(self, value: float) -> bool:
        return value > self.on if self.op == ">" else value < self.on


@dataclass
class LookConfig:
    """Rate-control camera settings (angles in the source's units, normally degrees)."""
    yaw_source: str = "face.yaw"
    pitch_source: str = "face.pitch"
    deadzone_deg: float = 6.0
    gain_px_per_deg_s: float = 30.0
    pitch_gain_px_per_deg_s: float | None = None    # None = same as yaw gain
    expo: float = 0.3                                # 0 = linear, 1 = fully quadratic
    expo_ref_deg: float = 15.0                       # angle beyond deadzone where expo is neutral
    max_px_per_frame: float = 40.0
    invert_y: bool = False
    mode: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "LookConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"look: unknown keys {sorted(unknown)}")
        cfg = cls(**d)
        _validate_source(cfg.yaw_source, "look.yaw_source")
        _validate_source(cfg.pitch_source, "look.pitch_source")
        if not 0.0 <= cfg.expo <= 1.0:
            raise ValueError("look.expo must be in [0, 1]")
        if cfg.deadzone_deg < 0 or cfg.max_px_per_frame <= 0 or cfg.expo_ref_deg <= 0:
            raise ValueError("look: deadzone >= 0, max_px_per_frame > 0, expo_ref_deg > 0")
        return cfg

    def speed(self, angle: float, gain: float) -> float:
        """Signed camera speed in px/s for an angle offset from neutral.

        ``e = |angle| - deadzone`` (0 inside the deadzone);
        ``speed = gain * e * ((1 - expo) + expo * e / expo_ref_deg)`` – linear for
        ``expo = 0``; with expo > 0 small offsets are gentler (fine aiming) while the
        speed at ``expo_ref_deg`` is unchanged.
        """
        e = abs(angle) - self.deadzone_deg
        if e <= 0:
            return 0.0
        mag = gain * e * ((1.0 - self.expo) + self.expo * e / self.expo_ref_deg)
        return math.copysign(mag, angle)


@dataclass
class ModeToggle:
    trigger: HoldTrigger
    modes: list[str]
    initial: str


@dataclass
class Profile:
    """A parsed, validated control profile."""
    name: str
    bindings: list[Binding]
    look: LookConfig | None
    pause: HoldTrigger | None
    mode_toggle: ModeToggle | None = None
    gestures: dict[str, dict[str, float]] = field(
        default_factory=lambda: copy.deepcopy(GESTURE_DEFAULTS))
    bindings_keys: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_KEYMAP))
    description: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        """Load and validate a profile YAML file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=_ProfileLoader) or {}
        return cls.from_dict(data, default_name=path.stem)

    @classmethod
    def from_dict(cls, d: dict, default_name: str = "profile") -> "Profile":
        """Build a profile from an already-parsed mapping (handy for tests)."""
        bindings = [Binding.from_dict(b, i) for i, b in enumerate(d.get("bindings") or [])]
        look = LookConfig.from_dict(d["look"]) if d.get("look") else None
        pause = HoldTrigger.from_dict(d["pause"], "pause") if d.get("pause") else None

        mode_toggle = None
        if d.get("mode_toggle"):
            mt = d["mode_toggle"]
            modes = list(mt.get("modes") or ["walk", "look"])
            if len(modes) < 2:
                raise ValueError("mode_toggle.modes needs at least two modes")
            initial = mt.get("initial", modes[0])
            if initial not in modes:
                raise ValueError(f"mode_toggle.initial {initial!r} not in {modes}")
            mode_toggle = ModeToggle(HoldTrigger.from_dict(mt, "mode_toggle"), modes, initial)
        valid_modes = set(mode_toggle.modes) if mode_toggle else set()
        tagged = [b.mode for b in bindings if b.mode] + ([look.mode] if look and look.mode else [])
        for m in tagged:
            if m not in valid_modes:
                raise ValueError(f"mode {m!r} used but mode_toggle.modes is {sorted(valid_modes)}")

        gestures = copy.deepcopy(GESTURE_DEFAULTS)
        for gname, overrides in (d.get("gestures") or {}).items():
            if gname not in gestures:
                raise ValueError(f"gestures: unknown gesture {gname!r}")
            unknown = set(overrides) - set(gestures[gname])
            if unknown:
                raise ValueError(f"gestures.{gname}: unknown keys {sorted(unknown)}")
            gestures[gname].update({k: float(v) for k, v in overrides.items()})

        keys = dict(DEFAULT_KEYMAP)
        keys.update({str(k): str(v) for k, v in (d.get("bindings_keys") or {}).items()})
        return cls(name=str(d.get("name", default_name)), bindings=bindings, look=look,
                   pause=pause, mode_toggle=mode_toggle, gestures=gestures,
                   bindings_keys=keys, description=str(d.get("description", "")))


class _Gestures:
    """Derives the ``gesture.*`` 0/1 values from the flat feature dict."""

    def __init__(self, cfg: dict[str, dict[str, float]]) -> None:
        self.curl = CurlStateMachine(**cfg["fist"])
        p = cfg["pinch"]
        self.pinch = Hysteresis(p["on"], p["off"], "<", p["min_hold_ms"])
        self.pinch_max_curl = p["max_curl_mean"]
        fp = cfg["finger_pinch"]
        self.finger_pinch = {n: Hysteresis(fp["on"], fp["off"], "<", fp["min_hold_ms"])
                             for n in FINGER_PINCHES}
        self.finger_pinch_max_curl = fp["max_curl_mean"]
        o = cfg["open_palm"]
        self.open_hand = Hysteresis(o["on"], o["off"], "<")
        self.open_off = Hysteresis(o["on"], o["off"], "<")
        self.open_min_pinch = o["min_pinch"]
        self.blink = Hysteresis(cfg["both_blink"]["on"], cfg["both_blink"]["off"], ">")
        self.smile = Hysteresis(cfg["smile"]["on"], cfg["smile"]["off"], ">")

    def _open(self, det: Hysteresis, flat: dict, prefix: str, t: float) -> bool:
        if not flat.get(f"{prefix}.present", 0.0):
            det.reset()
            return False
        # Treat a pinched thumb as "not open" by feeding a fully-curled value.
        curl = flat.get(f"{prefix}.curl_mean", 0.0)
        if flat.get(f"{prefix}.pinch", 1.0) < self.open_min_pinch:
            curl = 1.0
        return det.update(curl, t)

    def update(self, flat: dict, t: float) -> dict[str, float]:
        g: dict[str, float] = {}
        hand = bool(flat.get("hand.present", 0.0))
        if hand:
            curl_mean = flat.get("hand.curl_mean", 0.0)
            self.curl.update(curl_mean, t)
            # Thumb-tip distance to each fingertip; only the nearest one may count as a pinch.
            dists = {"index": flat.get("hand.pinch", 9.0)}
            dists.update({n: flat.get(f"hand.pinch_{n}", 9.0) for n in FINGER_PINCHES})
            nearest = min(dists, key=lambda k: dists[k])
            # A fist also closes thumb onto index; don't let that read as a pinch.
            ok = curl_mean < self.pinch_max_curl and nearest == "index"
            g["gesture.pinch"] = float(self.pinch.update(dists["index"] if ok else 1e9, t))
            for n in FINGER_PINCHES:
                ok = curl_mean < self.finger_pinch_max_curl and nearest == n
                g[f"gesture.pinch_{n}"] = float(self.finger_pinch[n].update(dists[n] if ok else 1e9, t))
        else:
            self.curl.reset()
            self.pinch.reset()
            g["gesture.pinch"] = 0.0
            for n in FINGER_PINCHES:
                self.finger_pinch[n].reset()
                g[f"gesture.pinch_{n}"] = 0.0
        g["gesture.fist"] = float(self.curl.is_closed)
        hand_open = self._open(self.open_hand, flat, "hand", t)
        off_open = self._open(self.open_off, flat, "offhand", t)
        g["gesture.open_palm"] = float(hand_open)
        g["gesture.both_open"] = float(hand_open and off_open)

        if flat.get("face.present", 0.0):
            blink = min(flat.get("face.blend.eyeBlinkLeft", 0.0),
                        flat.get("face.blend.eyeBlinkRight", 0.0))
            smile = min(flat.get("face.blend.mouthSmileLeft", 0.0),
                        flat.get("face.blend.mouthSmileRight", 0.0))
            g["gesture.both_blink"] = float(self.blink.update(blink, t))
            g["gesture.smile"] = float(self.smile.update(smile, t))
        else:
            self.blink.reset()
            self.smile.reset()
            g["gesture.both_blink"] = g["gesture.smile"] = 0.0
        return g


class Mapper:
    """Stateful profile interpreter: ``update(flat, t) -> Commands`` once per frame."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self._gestures = _Gestures(profile.gestures)
        self._detectors = [b.make_detector() for b in profile.bindings]
        self._tap_on = [False] * len(profile.bindings)            # tap bindings: last state
        self._tap_next: list[float | None] = [None] * len(profile.bindings)   # next repeat time
        self._pause_timer = HoldTimer(profile.pause.hold_s) if profile.pause else None
        mt = profile.mode_toggle
        self._mode_timer = HoldTimer(mt.trigger.hold_s) if mt else None
        self.mode: str | None = mt.initial if mt else None
        self.paused = False
        self._last_t: float | None = None

    # -- external control (e.g. F8 hotkey in the app) ---------------------------------
    def set_paused(self, paused: bool) -> None:
        """Pause or resume; all binding detectors restart from OFF."""
        self.paused = bool(paused)
        for det in self._detectors:
            det.reset()
        self._tap_on = [False] * len(self._detectors)
        self._tap_next = [None] * len(self._detectors)

    def toggle_pause(self) -> None:
        self.set_paused(not self.paused)

    # -- helpers -----------------------------------------------------------------------
    @staticmethod
    def _resolve(values: dict, source: str) -> tuple[float, bool]:
        """Return ``(value, present)`` for a source; missing keys read as 0."""
        pk = presence_key(source)
        present = True if pk is None else bool(values.get(pk, 0.0))
        return float(values.get(source, 0.0)), present

    def _trigger(self, trig: HoldTrigger, timer: HoldTimer, values: dict, t: float) -> bool:
        value, present = self._resolve(values, trig.source)
        return timer.update(present and trig.cond(value), t)

    def _look(self, values: dict, dt: float) -> tuple[float, float, dict]:
        look = self.profile.look
        if look is None:
            return 0.0, 0.0, {"active": False}
        yaw, yaw_ok = self._resolve(values, look.yaw_source)
        pitch, pitch_ok = self._resolve(values, look.pitch_source)
        live = yaw_ok and pitch_ok and (look.mode is None or look.mode == self.mode)
        if not live:
            return 0.0, 0.0, {"active": False, "yaw_px_s": 0.0, "pitch_px_s": 0.0}
        pitch_gain = look.pitch_gain_px_per_deg_s or look.gain_px_per_deg_s
        vx = look.speed(yaw, look.gain_px_per_deg_s)
        # Pitch + = looking up, but screen y grows downward -> negate (unless inverted).
        vy = look.speed(pitch, pitch_gain) * (1.0 if look.invert_y else -1.0)
        cap = look.max_px_per_frame
        dx = max(-cap, min(cap, vx * dt))
        dy = max(-cap, min(cap, vy * dt))
        return dx, dy, {"active": True, "yaw_px_s": vx, "pitch_px_s": vy}

    # -- main entry point ----------------------------------------------------------------
    def update(self, flat: dict[str, float], t: float) -> Commands:
        """Turn one frame of calibrated features into :class:`Commands`."""
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), MAX_DT_S)
        self._last_t = t
        values = dict(flat)
        values.update(self._gestures.update(flat, t))
        debug: dict[str, Any] = {
            "dt": dt,
            "curl_state": self._gestures.curl.state,
            "curl_theta": self._gestures.curl.theta,
            "gestures": {k: v for k, v in values.items() if k.startswith("gesture.")},
        }

        # Pause gesture is watched even while paused, so the user can resume.
        if self._pause_timer is not None and self._trigger(
                self.profile.pause, self._pause_timer, values, t):
            self.toggle_pause()
        debug["pause_progress"] = self._pause_timer.progress if self._pause_timer else 0.0
        debug["paused"] = self.paused
        if self.paused:
            debug["mode"] = self.mode
            cmds = Commands.idle(paused=True)
            cmds.debug = debug
            return cmds

        mt = self.profile.mode_toggle
        if mt is not None and self._trigger(mt.trigger, self._mode_timer, values, t):
            self.mode = mt.modes[(mt.modes.index(self.mode) + 1) % len(mt.modes)]
        debug["mode"] = self.mode
        debug["mode_progress"] = self._mode_timer.progress if self._mode_timer else 0.0

        # Evaluate every binding; an action is on if any of its bindings is on.
        drive: dict[str, float] = {}
        taps: list[str] = []
        bdebug = []
        for i, (b, det) in enumerate(zip(self.profile.bindings, self._detectors)):
            value, present = self._resolve(values, b.source)
            if not present or (b.mode is not None and b.mode != self.mode):
                det.reset()
                on = False
            else:
                on = det.update(value, t)
            if b.action in TAP_ACTIONS:
                # One-shot: fire on the rising edge, then every repeat_ms while still held.
                fire = False
                if on and not self._tap_on[i]:
                    fire = True
                    self._tap_next[i] = t + b.repeat_ms / 1000.0 if b.repeat_ms > 0 else None
                elif on and self._tap_next[i] is not None and t >= self._tap_next[i]:
                    fire = True
                    self._tap_next[i] = t + b.repeat_ms / 1000.0
                if not on:
                    self._tap_next[i] = None
                self._tap_on[i] = on
                if fire and b.action not in taps:
                    taps.append(b.action)
                bdebug.append({"action": b.action, "source": b.source, "value": value,
                               "present": present, "on": on})
                continue
            if on:
                drive[b.action] = max(drive.get(b.action, -math.inf), b.drive(value))
            bdebug.append({"action": b.action, "source": b.source, "value": value,
                           "present": present, "on": on})
        debug["bindings"] = bdebug

        for a, b in OPPOSING:
            if a in drive and b in drive:
                del drive[b if drive[a] >= drive[b] else a]

        dx, dy, look_dbg = self._look(values, dt)
        debug["look"] = look_dbg
        return Commands(
            keys={a for a in drive if a in KEY_ACTIONS},
            mouse_dx=dx,
            mouse_dy=dy,
            left_button="left_button" in drive,
            right_button="right_button" in drive,
            taps=taps,
            debug=debug,
        )
