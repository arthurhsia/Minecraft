"""Neutral-pose calibration: capture, robust estimation, JSON persistence.

Calibration flow (driven by the app / HUD):

1. "Sit relaxed, look at the screen, rest your control hand" – feed raw
   :class:`~mcbio.types.Features` into a :class:`Calibrator` for ~2 s; ``result()``
   returns a :class:`~mcbio.types.Calibration` made of per-signal **medians**, so a few
   bad tracking frames (or a blink) do not shift the neutral.
2. Optional: "move your hand to a comfortable maximum in every direction" – pass those
   frames to :func:`capture_range` to set ``palm_range`` (full joystick deflection).
3. :func:`save` / :func:`load` the result as JSON (default ``~/.mcbio/calibration.json``).

Calibration needs *raw* tracker values (the flattened dict is already calibrated),
which is why everything here consumes ``Features`` rather than ``flatten()`` output.
"""
from __future__ import annotations

import dataclasses
import json
import math
import statistics
from pathlib import Path

from .types import Calibration, Features, HandFeatures

DEFAULT_PATH = Path("~/.mcbio/calibration.json").expanduser()
DEFAULT_PALM_RANGE = 0.12
MIN_PALM_RANGE = 0.04       # below this the joystick becomes twitchy
MAX_PALM_RANGE = 0.5        # half the image width


class CalibrationError(RuntimeError):
    """Raised when the captured samples cannot produce a calibration."""


def dominant(hands: list[HandFeatures], dominant_hand: str) -> HandFeatures | None:
    """Pick the dominant hand: matching handedness first, else the only/first hand."""
    present = [h for h in hands if h.present]
    for h in present:
        if h.handedness == dominant_hand:
            return h
    return present[0] if present else None


def calibrate(
    samples: list[Features],
    dominant_hand: str = "Right",
    palm_range: float = DEFAULT_PALM_RANGE,
) -> Calibration:
    """Robust (median) neutral pose from samples captured while the user is still.

    Face fields come from frames where the face was tracked, hand fields from frames
    where a hand was tracked; whichever is missing keeps the :class:`Calibration`
    default (so face-only and hand-only users can still calibrate).
    """
    faces = [s.face for s in samples if s.face.present]
    palms = [h for h in (dominant(s.hands, dominant_hand) for s in samples) if h]
    if not faces and not palms:
        raise CalibrationError("no face or hand was tracked during calibration")

    calib = Calibration(dominant_hand=dominant_hand, palm_range=palm_range)
    med = statistics.median
    if faces:
        calib.neutral_yaw = med(f.head.yaw for f in faces)
        calib.neutral_pitch = med(f.head.pitch for f in faces)
        calib.neutral_roll = med(f.head.roll for f in faces)
        calib.neutral_nose = (med(f.nose[0] for f in faces), med(f.nose[1] for f in faces))
        calib.neutral_face_scale = med(f.scale for f in faces)
        names = sorted({n for f in faces for n in f.blend})
        # A blendshape missing in a frame counts as 0 for that frame.
        calib.neutral_blend = {n: med(f.blend.get(n, 0.0) for f in faces) for n in names}
    if palms:
        calib.neutral_palm = (med(h.palm[0] for h in palms), med(h.palm[1] for h in palms))
    return calib


def capture_range(
    samples: list[Features],
    calib: Calibration,
    percentile: float = 90.0,
    default: float = DEFAULT_PALM_RANGE,
) -> Calibration:
    """Return ``calib`` with ``palm_range`` set from a "reach comfortably" capture.

    ``palm_range`` is the ``percentile``-th percentile of the dominant palm's distance
    from ``calib.neutral_palm`` (a high percentile rather than the max, so one wild
    frame cannot inflate it), clamped to [MIN_PALM_RANGE, MAX_PALM_RANGE].  Without any
    hand samples it falls back to ``default``.
    """
    nx, ny = calib.neutral_palm
    dists = sorted(
        math.hypot(h.palm[0] - nx, h.palm[1] - ny)
        for h in (dominant(s.hands, calib.dominant_hand) for s in samples) if h
    )
    if not dists:
        return dataclasses.replace(calib, palm_range=default)
    # Nearest-rank percentile.
    k = min(len(dists) - 1, max(0, math.ceil(percentile / 100.0 * len(dists)) - 1))
    rng = min(MAX_PALM_RANGE, max(MIN_PALM_RANGE, dists[k]))
    return dataclasses.replace(calib, palm_range=rng)


class Calibrator:
    """Accumulates frames for ``duration_s`` seconds, then yields a Calibration.

    Usage::

        cal = Calibrator()
        while not cal.add(features):   # returns True once enough time has passed
            show_progress(cal.progress)
        calib = cal.result()
    """

    def __init__(self, duration_s: float = 2.0, dominant_hand: str = "Right",
                 min_samples: int = 10) -> None:
        self.duration_s = duration_s
        self.dominant_hand = dominant_hand
        self.min_samples = min_samples
        self.samples: list[Features] = []

    @property
    def progress(self) -> float:
        """0..1 fraction of the capture window covered so far."""
        if not self.samples:
            return 0.0
        if self.duration_s <= 0:
            return 1.0
        return min(1.0, (self.samples[-1].t - self.samples[0].t) / self.duration_s)

    @property
    def done(self) -> bool:
        return self.progress >= 1.0 and len(self.samples) >= self.min_samples

    def add(self, features: Features) -> bool:
        """Add one frame; return True when the capture window is complete."""
        self.samples.append(features)
        return self.done

    def result(self, palm_range: float = DEFAULT_PALM_RANGE) -> Calibration:
        if len(self.samples) < self.min_samples:
            raise CalibrationError(
                f"only {len(self.samples)} frames captured, need {self.min_samples}")
        return calibrate(self.samples, self.dominant_hand, palm_range)


def save(calib: Calibration, path: str | Path = DEFAULT_PATH) -> Path:
    """Write ``calib`` as JSON (creating the folder); returns the path written."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(calib), indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def load(path: str | Path = DEFAULT_PATH) -> Calibration:
    """Read a Calibration JSON. Unknown keys are ignored, missing ones keep defaults.

    Raises ``FileNotFoundError`` if the file does not exist (the app then runs the
    calibration step).
    """
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    fields = {f.name for f in dataclasses.fields(Calibration)}
    kwargs = {k: v for k, v in data.items() if k in fields}
    for key in ("neutral_nose", "neutral_palm"):        # JSON lists -> tuples
        if key in kwargs:
            kwargs[key] = tuple(float(v) for v in kwargs[key])
    if "neutral_blend" in kwargs:
        kwargs["neutral_blend"] = {str(k): float(v) for k, v in kwargs["neutral_blend"].items()}
    return Calibration(**kwargs)
