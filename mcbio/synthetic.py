"""Synthetic feature generator: a believable scripted "user" with no camera or mediapipe.

``SyntheticFeatures`` produces a stream of :class:`mcbio.types.Features` at a fixed frame
rate that looks like what the real tracker would report for a person who:

* sits still (neutral) while switching to Minecraft,
* pushes their hand forward / back / sideways (move),
* turns their head left / right (look),
* makes a fist (attack), pinches (interact), opens the mouth (jump), raises brows (sneak),
* shows both open palms for ~1.5 s (pause toggle), then does it again (resume).

Values are smoothed with a first-order lag (people don't teleport) and carry a little
Gaussian sensor noise, so the downstream hysteresis / debouncing is exercised for real.

Two modes:

``mode="script"``     loops the scripted session above (default; used by ``--synthetic``).
``mode="responsive"`` idles at neutral and *reacts* to protocol cues passed through
                      :meth:`SyntheticFeatures.cue`, with a human-like reaction time,
                      occasional wrong gestures and rare false activations during REST.
                      This lets ``--protocol --synthetic`` generate realistic experiment
                      data for testing ``scripts/analyze_session.py``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator

from mcbio.types import Calibration, FaceFeatures, Features, HandFeatures, HeadPose

# --------------------------------------------------------------------------------------
# The synthetic user's neutral pose (deliberately not zero, so calibration subtraction
# is actually tested).
# --------------------------------------------------------------------------------------
NEUTRAL: dict[str, float] = {
    "yaw": 2.0, "pitch": -4.0, "roll": 1.0,
    "nose_x": 0.50, "nose_y": 0.42, "face_scale": 0.085,
    "palm_x": 0.66, "palm_y": 0.62, "palm_scale": 0.11,
    "thumb": 0.20, "index": 0.25, "middle": 0.25, "ring": 0.28, "little": 0.30,
    "pinch": 0.90,
    "hand": 1.0,               # dominant hand present (0/1, not smoothed)
    "off": 0.0,                # off-hand present (0/1, not smoothed)
    "off_x": 0.30, "off_y": 0.60, "off_curl": 0.35,
    "jawOpen": 0.04, "browInnerUp": 0.08, "mouthPucker": 0.05, "cheekPuff": 0.02,
    "mouthSmileLeft": 0.05, "mouthSmileRight": 0.05,
    "eyeBlinkLeft": 0.06, "eyeBlinkRight": 0.06,
}
PALM_RANGE = 0.12            # matches Calibration.palm_range used by calibration()

#: Gesture name -> overrides of NEUTRAL (absolute target values).
GESTURES: dict[str, dict[str, float]] = {
    "neutral": {},
    "look_left": {"yaw": NEUTRAL["yaw"] - 20.0},
    "look_right": {"yaw": NEUTRAL["yaw"] + 20.0},
    "look_up": {"pitch": NEUTRAL["pitch"] + 14.0},
    "look_down": {"pitch": NEUTRAL["pitch"] - 14.0},
    # palm y decreases = hand moves UP in the image = "forward" (flatten flips y)
    "forward": {"palm_y": NEUTRAL["palm_y"] - 0.085},
    "back": {"palm_y": NEUTRAL["palm_y"] + 0.10},
    "strafe_left": {"palm_x": NEUTRAL["palm_x"] - 0.11},
    "strafe_right": {"palm_x": NEUTRAL["palm_x"] + 0.11},
    "fist": {"thumb": 0.65, "index": 0.92, "middle": 0.94, "ring": 0.93, "little": 0.90},
    "pinch": {"pinch": 0.08, "thumb": 0.40, "index": 0.35},
    "mouth_open": {"jawOpen": 0.75},
    "brow_raise": {"browInnerUp": 0.80},
    "pucker": {"mouthPucker": 0.75},
    "both_palms": {"off": 1.0, "off_curl": 0.03, "thumb": 0.02, "index": 0.02,
                   "middle": 0.02, "ring": 0.03, "little": 0.04, "pinch": 1.30,
                   "off_y": 0.58},
    "hand_away": {"hand": 0.0},
}

#: Default scripted session: (gesture, nominal duration in seconds).
#: The first 3.2 s are neutral, matching the app's start-up countdown.
DEFAULT_SCRIPT: list[tuple[str, float]] = [
    ("neutral", 3.2),
    ("forward", 0.9), ("neutral", 0.3),
    ("look_left", 0.7), ("neutral", 0.3),
    ("look_right", 0.7), ("neutral", 0.3),
    ("strafe_left", 0.7), ("neutral", 0.3),
    ("strafe_right", 0.7), ("neutral", 0.3),
    ("back", 0.6), ("neutral", 0.3),
    ("fist", 0.9), ("neutral", 0.3),
    ("pinch", 0.6), ("neutral", 0.3),
    ("mouth_open", 0.7), ("neutral", 0.4),
    ("both_palms", 1.6), ("neutral", 1.0),     # -> pause
    ("forward", 0.8), ("neutral", 0.4),        # ignored: we are paused
    ("both_palms", 1.6), ("neutral", 3.2),     # -> resume (+ arming countdown)
    ("brow_raise", 0.6), ("neutral", 0.3),
    ("look_up", 0.6), ("look_down", 0.6), ("neutral", 0.4),
    ("hand_away", 1.0), ("neutral", 1.0),
]

#: Protocol cue label -> gesture the responsive synthetic user performs.
CUE_TO_GESTURE: dict[str, str] = {
    "FORWARD": "forward", "BACK": "back", "LEFT": "strafe_left", "RIGHT": "strafe_right",
    "JUMP": "mouth_open", "ATTACK": "fist", "INTERACT": "pinch",
    "LOOK LEFT": "look_left", "LOOK RIGHT": "look_right", "REST": "neutral",
}

#: Noise standard deviation per parameter (sensor jitter).
NOISE: dict[str, float] = {
    "yaw": 0.35, "pitch": 0.35, "roll": 0.3, "nose_x": 0.0015, "nose_y": 0.0015,
    "face_scale": 0.0005, "palm_x": 0.003, "palm_y": 0.003, "palm_scale": 0.001,
    "thumb": 0.015, "index": 0.015, "middle": 0.015, "ring": 0.015, "little": 0.015,
    "pinch": 0.02, "off_x": 0.003, "off_y": 0.003, "off_curl": 0.015,
}
_BINARY = ("hand", "off")


@dataclass
class _Segment:
    gesture: str
    t_start: float
    t_end: float


class SyntheticFeatures:
    """Deterministic (given ``seed``) generator of :class:`Features`.

    Iterate it (``for f in SyntheticFeatures(0): ...``) or call :meth:`next`. Time starts at
    ``t0`` and advances by ``1/fps`` per frame (simulated time, no sleeping).

    Args:
        seed: random seed (durations jitter, noise, responsive behaviour).
        fps: simulated frame rate.
        mode: ``"script"`` or ``"responsive"`` (see module docstring).
        script: optional custom list of ``(gesture, seconds)``; loops forever.
        tau: first-order lag time constant (s) for transitions.
        reaction_s: mean reaction time of the responsive user (s).
        error_rate: probability the responsive user does the wrong gesture.
        false_rate_per_s: rate of spurious gestures during REST/idle (responsive mode).
        t0: start time.
    """

    def __init__(self, seed: int = 0, fps: float = 30.0, mode: str = "script",
                 script: list[tuple[str, float]] | None = None, tau: float = 0.08,
                 reaction_s: float | None = None, error_rate: float = 0.06,
                 false_rate_per_s: float = 0.02, t0: float = 0.0) -> None:
        if mode not in ("script", "responsive"):
            raise ValueError(f"unknown mode {mode!r}")
        self.rng = random.Random(seed)
        self.fps = float(fps)
        self.mode = mode
        self.tau = tau
        # every synthetic "subject" gets a personal mean reaction time
        self.reaction_s = reaction_s if reaction_s is not None else self.rng.uniform(0.40, 0.65)
        self.error_rate = error_rate
        self.false_rate_per_s = false_rate_per_s
        self.t0 = t0
        self.frame_index = 0
        self.state: dict[str, float] = dict(NEUTRAL)
        self._script = list(script or DEFAULT_SCRIPT)
        self._script_pos = 0
        self._segments: list[_Segment] = []      # scheduled segments (time ordered)
        self._script_end = t0
        if mode == "script":
            self._extend_script(t0 + 60.0)

    # ------------------------------------------------------------------ public API
    @property
    def t(self) -> float:
        """Time of the *next* frame to be generated."""
        return self.t0 + self.frame_index / self.fps

    @property
    def segment(self) -> str:
        """Name of the gesture currently being performed (for debugging/HUD)."""
        return self._gesture_at(self.t)

    def calibration(self) -> Calibration:
        """The Calibration matching this synthetic user's neutral pose."""
        n = NEUTRAL
        return Calibration(
            neutral_yaw=n["yaw"], neutral_pitch=n["pitch"], neutral_roll=n["roll"],
            neutral_nose=(n["nose_x"], n["nose_y"]), neutral_face_scale=n["face_scale"],
            neutral_palm=(n["palm_x"], n["palm_y"]), palm_range=PALM_RANGE,
            neutral_blend={k: n[k] for k in ("jawOpen", "browInnerUp", "mouthPucker",
                                             "cheekPuff", "mouthSmileLeft", "mouthSmileRight",
                                             "eyeBlinkLeft", "eyeBlinkRight")},
            dominant_hand="Right",
        )

    def cue(self, label: str, t: float) -> None:
        """Tell the responsive user a protocol cue appeared at time ``t``.

        Schedules the matching gesture after a human-like reaction time (log-normal-ish,
        ~``reaction_s`` mean), held for ~0.8 s. With probability ``error_rate`` a wrong
        gesture is performed instead. REST cues schedule nothing.
        """
        if self.mode != "responsive":
            return
        label = label.upper()
        gesture = CUE_TO_GESTURE.get(label, "neutral")
        if gesture == "neutral":
            return
        if self.rng.random() < self.error_rate:
            others = [g for g in CUE_TO_GESTURE.values() if g not in ("neutral", gesture)]
            gesture = self.rng.choice(others)
        rt = max(0.18, self.rng.gauss(self.reaction_s, 0.10 + 0.15 * self.reaction_s))
        hold = self.rng.uniform(0.6, 1.0)
        start = t + rt
        # a new cue cancels anything still scheduled in the future
        self._segments = [s for s in self._segments if s.t_end <= t]
        self._segments.append(_Segment(gesture, start, start + hold))

    def next(self) -> Features:
        """Generate the next frame."""
        t = self.t
        dt = 1.0 / self.fps
        if self.mode == "script" and t + 5.0 > self._script_end:
            self._extend_script(t + 60.0)
        if self.mode == "responsive":
            self._maybe_false_activation(t, dt)
            self._segments = [s for s in self._segments if s.t_end > t - 1.0]
        target = dict(NEUTRAL)
        target.update(GESTURES.get(self._gesture_at(t), {}))
        alpha = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        for k, goal in target.items():
            if k in _BINARY:
                self.state[k] = goal
            else:
                self.state[k] += alpha * (goal - self.state[k])
        self.frame_index += 1
        return self._make_features(t)

    def __iter__(self) -> Iterator[Features]:
        return self

    def __next__(self) -> Features:
        return self.next()

    # ------------------------------------------------------------------ internals
    def _extend_script(self, until: float) -> None:
        while self._script_end < until:
            g, dur = self._script[self._script_pos % len(self._script)]
            self._script_pos += 1
            # don't jitter the opening neutral (it must outlast the start countdown)
            jitter = 1.0 if self._script_pos == 1 else self.rng.uniform(0.85, 1.15)
            d = dur * jitter
            self._segments.append(_Segment(g, self._script_end, self._script_end + d))
            self._script_end += d
        # drop segments far in the past
        self._segments = [s for s in self._segments if s.t_end > self.t - 2.0]

    def _gesture_at(self, t: float) -> str:
        for s in self._segments:
            if s.t_start <= t < s.t_end:
                return s.gesture
        return "neutral"

    def _maybe_false_activation(self, t: float, dt: float) -> None:
        busy = any(s.t_end > t for s in self._segments)
        if not busy and self.rng.random() < self.false_rate_per_s * dt:
            g = self.rng.choice(["forward", "look_left", "mouth_open", "fist", "strafe_right"])
            self._segments.append(_Segment(g, t, t + self.rng.uniform(0.3, 0.6)))

    def _noisy(self, k: str) -> float:
        return self.state[k] + self.rng.gauss(0.0, NOISE.get(k, 0.0))

    def _make_features(self, t: float) -> Features:
        s = self.state
        clip01 = lambda v: min(1.0, max(0.0, v))  # noqa: E731
        head = HeadPose(yaw=self._noisy("yaw"), pitch=self._noisy("pitch"), roll=self._noisy("roll"))
        blend = {k: clip01(s[k] + self.rng.gauss(0.0, 0.01)) for k in
                 ("jawOpen", "browInnerUp", "mouthPucker", "cheekPuff", "mouthSmileLeft",
                  "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight")}
        nose = (self._noisy("nose_x"), self._noisy("nose_y"))
        face = FaceFeatures(present=True, head=head, blend=blend, nose=nose,
                            scale=self._noisy("face_scale"),
                            landmarks=face_landmarks(nose, s["face_scale"], head.yaw - NEUTRAL["yaw"],
                                                     head.pitch - NEUTRAL["pitch"], blend["jawOpen"]))
        hands: list[HandFeatures] = []
        if s["hand"] >= 0.5:
            curl = tuple(clip01(self._noisy(k)) for k in ("thumb", "index", "middle", "ring", "little"))
            palm = (self._noisy("palm_x"), self._noisy("palm_y"))
            scale = self._noisy("palm_scale")
            hands.append(HandFeatures(present=True, handedness="Right", palm=palm,
                                      curl=curl,  # type: ignore[arg-type]
                                      pinch=max(0.0, self._noisy("pinch")), scale=scale,
                                      landmarks=hand_landmarks(palm, scale, curl, "Right")))
        if s["off"] >= 0.5:
            c = clip01(self._noisy("off_curl"))
            curl = (c, c, c, c, c)
            palm = (self._noisy("off_x"), self._noisy("off_y"))
            hands.append(HandFeatures(present=True, handedness="Left", palm=palm, curl=curl,
                                      pinch=1.2, scale=s["palm_scale"],
                                      landmarks=hand_landmarks(palm, s["palm_scale"], curl, "Left")))
        return Features(t=t, face=face, hands=hands)


# --------------------------------------------------------------------------------------
# Cheap 2-D landmark synthesis so the HUD has something to draw in synthetic mode.
# --------------------------------------------------------------------------------------
def hand_landmarks(palm: tuple[float, float], scale: float,
                   curl: tuple[float, ...], handedness: str = "Right") -> list[tuple[float, float, float]]:
    """Build 21 MediaPipe-ordered hand landmarks (mirrored normalized coords).

    Palm faces the camera, fingers point up; each finger joint bends by ``curl * 75°``
    toward the palm (foreshortened in 2-D by shrinking the segment). The thumb sits on
    the body-centre side: screen-left for the user's right hand in the mirrored view.
    """
    px, py = palm
    side = -1.0 if handedness == "Right" else 1.0
    s = max(scale, 1e-3)
    wrist = (px, py + 0.55 * s)
    pts: list[tuple[float, float, float]] = [(wrist[0], wrist[1], 0.0)]
    # (base offset x, base offset y, direction angle from vertical, segment lengths)
    fingers = [
        (side * 0.35, 0.30, side * 0.9, (0.35, 0.30, 0.25)),   # thumb
        (side * 0.28, -0.35, side * 0.12, (0.40, 0.25, 0.20)),  # index
        (side * 0.05, -0.42, 0.0, (0.45, 0.28, 0.22)),          # middle
        (-side * 0.18, -0.38, -side * 0.10, (0.40, 0.25, 0.20)),  # ring
        (-side * 0.38, -0.28, -side * 0.22, (0.32, 0.20, 0.17)),  # little
    ]
    for (bx, by, ang, segs), c in zip(fingers, curl):
        x, y = px + bx * s, py + by * s
        pts.append((x, y, 0.0))
        for k, L in enumerate(segs):
            # Seen from the front, a curling finger folds down over the palm: the
            # vertical component of each successive segment rotates by up to ~70°/joint.
            theta = min(math.pi, c * (k + 1) * 1.2)
            x += math.sin(ang) * L * s
            y += -math.cos(ang) * math.cos(theta) * L * s
            pts.append((x, y, -0.02 * c * (k + 1)))
    return pts


def face_landmarks(nose: tuple[float, float], scale: float, dyaw: float, dpitch: float,
                   jaw: float) -> list[tuple[float, float, float]]:
    """A small face point subset: nose, eyes, mouth corners, upper/lower lip, chin."""
    nx, ny = nose
    s = max(scale, 1e-3)
    ox = math.sin(math.radians(dyaw)) * 0.8 * s     # head turn shifts features sideways
    oy = -math.sin(math.radians(dpitch)) * 0.8 * s
    pts = [
        (nx, ny),                                     # nose tip
        (nx - 0.5 * s + 0.4 * ox, ny - 0.45 * s + 0.4 * oy),  # eye L
        (nx + 0.5 * s + 0.4 * ox, ny - 0.45 * s + 0.4 * oy),  # eye R
        (nx - 0.38 * s - 0.2 * ox, ny + 0.55 * s - 0.2 * oy),  # mouth L
        (nx + 0.38 * s - 0.2 * ox, ny + 0.55 * s - 0.2 * oy),  # mouth R
        (nx - 0.2 * ox, ny + 0.48 * s - 0.2 * oy),             # upper lip
        (nx - 0.2 * ox, ny + (0.62 + 0.6 * jaw) * s - 0.2 * oy),  # lower lip
        (nx - 0.3 * ox, ny + (1.15 + 0.5 * jaw) * s - 0.3 * oy),  # chin
    ]
    return [(x, y, 0.0) for x, y in pts]
