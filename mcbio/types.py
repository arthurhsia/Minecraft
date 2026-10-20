"""Shared data types: the contract between tracking, mapping, outputs and the app.

Coordinate conventions
----------------------
* Image coordinates are MIRRORED (like a selfie view): x in [0,1] increases to the
  user's right, y in [0,1] increases downward.
* Head pose in degrees: yaw + = user turns head to their right; pitch + = looking up;
  roll + = tilting head toward the user's right shoulder.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HeadPose:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class HandFeatures:
    present: bool = False
    handedness: str = "Right"                       # as the USER sees it
    palm: tuple[float, float] = (0.5, 0.5)          # mirrored normalized image coords
    curl: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)  # thumb..little, 0 straight .. 1 curled
    pinch: float = 1.0                              # thumb-index tip distance / palm size
    scale: float = 0.0                              # palm size in image units
    landmarks: list | None = None                   # 21 x (x, y, z), mirrored normalized


@dataclass
class FaceFeatures:
    present: bool = False
    head: HeadPose = field(default_factory=HeadPose)
    blend: dict[str, float] = field(default_factory=dict)   # mediapipe blendshape scores 0..1
    nose: tuple[float, float] = (0.5, 0.5)
    scale: float = 0.0                              # face size in image units (inter-ocular distance)
    landmarks: list | None = None


@dataclass
class Features:
    t: float = 0.0                                  # monotonic seconds of the captured frame
    face: FaceFeatures = field(default_factory=FaceFeatures)
    hands: list[HandFeatures] = field(default_factory=list)


MOVE_KEYS = ("forward", "back", "left", "right", "jump", "sneak", "sprint")


@dataclass
class Commands:
    keys: set[str] = field(default_factory=set)     # held logical actions, names from MOVE_KEYS
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    left_button: bool = False
    right_button: bool = False
    taps: list[str] = field(default_factory=list)   # one-shot actions, e.g. "inventory", "hotbar_next"
    paused: bool = False
    debug: dict = field(default_factory=dict)

    @staticmethod
    def idle(paused: bool = False) -> "Commands":
        return Commands(paused=paused)


@dataclass
class Calibration:
    """User's neutral pose, captured by calibration.py and used by features.flatten().

    All "neutral_*" values are what the tracker reported while the user sat relaxed,
    looking at the screen, with their control hand resting in its comfortable spot.
    """
    neutral_yaw: float = 0.0
    neutral_pitch: float = 0.0
    neutral_roll: float = 0.0
    neutral_nose: tuple[float, float] = (0.5, 0.5)
    neutral_face_scale: float = 0.0
    neutral_palm: tuple[float, float] = (0.5, 0.6)
    palm_range: float = 0.12        # palm travel (image units) treated as "full stick deflection"
    neutral_blend: dict[str, float] = field(default_factory=dict)   # resting blendshape levels
    dominant_hand: str = "Right"
