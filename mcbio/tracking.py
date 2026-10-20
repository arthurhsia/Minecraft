"""Webcam capture and MediaPipe Tasks trackers (hand + face landmarkers, VIDEO mode).

``mediapipe`` and ``cv2`` are imported lazily inside constructors/methods so the
rest of the package (and the test-suite) imports without them or the model files.

Mirroring and handedness
------------------------
:class:`Camera` mirrors every frame (selfie view) *before* tracking, so x grows
toward the user's right in everything downstream. MediaPipe's hand model labels
handedness assuming exactly such a mirrored input, so its "Left"/"Right" already
names the user's real hand. If you construct a :class:`Tracker` for un-mirrored
frames, pass ``frames_mirrored=False`` and the labels are swapped.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .features import FeatureExtractor
from .types import Features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
DEFAULT_FACE_MODEL = MODELS_DIR / "face_landmarker.task"


def _check_model(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"MediaPipe model not found: {p}\n"
            "Download the models first with:  python scripts/download_models.py"
        )
    return p


class Camera:
    """OpenCV webcam capture returning mirrored BGR frames with monotonic timestamps.

    On macOS the AVFoundation backend is used (the default backend may pick a
    slower path); grant the terminal/IDE Camera permission in System Settings ->
    Privacy & Security -> Camera.
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30, mirror: bool = True) -> None:
        import cv2

        self._cv2 = cv2
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {index}. Check that no other app is using it and that "
                "Camera permission is granted to your terminal/IDE (macOS: System Settings -> "
                "Privacy & Security -> Camera)."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.mirror = mirror

    @property
    def size(self) -> tuple[int, int]:
        """Actual (width, height) delivered by the driver."""
        cv2 = self._cv2
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Grab one frame -> (ok, bgr_frame (mirrored if enabled), t_monotonic_seconds)."""
        ok, frame = self._cap.read()
        t = time.monotonic()
        if not ok or frame is None:
            return False, None, t
        if self.mirror:
            frame = self._cv2.flip(frame, 1)
        return True, frame, t

    def close(self) -> None:
        """Release the device (idempotent)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _VideoTask:
    """Shared plumbing for MediaPipe VIDEO-mode tasks: monotonic int-ms timestamps."""

    _landmarker: Any

    def __init__(self) -> None:
        self._last_ts_ms = -1

    def _timestamp_ms(self, t: float) -> int:
        # VIDEO mode rejects non-increasing timestamps; nudge duplicates forward by 1 ms.
        ts = int(round(t * 1000.0))
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def _detect(self, rgb: np.ndarray, t: float) -> Any:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb, dtype=np.uint8))
        return self._landmarker.detect_for_video(image, self._timestamp_ms(t))

    def close(self) -> None:
        """Free the native graph (idempotent)."""
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class HandTracker(_VideoTask):
    """MediaPipe ``HandLandmarker`` (Tasks API, VIDEO running mode).

    ``process(rgb, t)`` returns the raw ``HandLandmarkerResult`` (``hand_landmarks``,
    ``hand_world_landmarks``, ``handedness``). Handedness labels are correct for
    the user when the frame is mirrored (see module docstring).
    """

    def __init__(self, model_path: str | Path = DEFAULT_HAND_MODEL, num_hands: int = 2,
                 min_det: float = 0.5, min_track: float = 0.5) -> None:
        path = _check_model(model_path)
        super().__init__()
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_det,
            min_hand_presence_confidence=min_track,
            min_tracking_confidence=min_track,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_frame: np.ndarray, t: float) -> Any:
        """Run on an RGB uint8 frame captured at monotonic time ``t`` (seconds)."""
        return self._detect(rgb_frame, t)


class FaceTracker(_VideoTask):
    """MediaPipe ``FaceLandmarker`` (Tasks API, VIDEO running mode), one face.

    ``process(rgb, t)`` returns the raw ``FaceLandmarkerResult`` (478 landmarks,
    optional blendshapes and 4x4 facial transformation matrix).
    """

    def __init__(self, model_path: str | Path = DEFAULT_FACE_MODEL, output_blendshapes: bool = True,
                 output_transform: bool = True, min_det: float = 0.5, min_track: float = 0.5) -> None:
        path = _check_model(model_path)
        super().__init__()
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_det,
            min_face_presence_confidence=min_track,
            min_tracking_confidence=min_track,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=output_transform,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame: np.ndarray, t: float) -> Any:
        """Run on an RGB uint8 frame captured at monotonic time ``t`` (seconds)."""
        return self._detect(rgb_frame, t)


class Tracker:
    """Facade: BGR frame -> smoothed :class:`Features` using hand and/or face trackers.

    The most recent raw MediaPipe results are kept in ``last_hand_result`` /
    ``last_face_result`` (e.g. for the HUD).
    """

    def __init__(self, hand_model: str | Path = DEFAULT_HAND_MODEL, face_model: str | Path = DEFAULT_FACE_MODEL,
                 use_hands: bool = True, use_face: bool = True, num_hands: int = 2,
                 frames_mirrored: bool = True, extractor: FeatureExtractor | None = None) -> None:
        self.hands = HandTracker(hand_model, num_hands=num_hands) if use_hands else None
        try:
            self.face = FaceTracker(face_model) if use_face else None
        except Exception:
            if self.hands is not None:
                self.hands.close()
            raise
        self.extractor = extractor or FeatureExtractor()
        # MediaPipe handedness assumes a mirrored image; swap only for raw frames.
        self.extractor.swap_handedness = not frames_mirrored
        self.last_hand_result: Any = None
        self.last_face_result: Any = None

    def process(self, bgr: np.ndarray, t: float) -> Features:
        """Track one (mirrored) BGR frame captured at monotonic time ``t``."""
        import cv2

        h, w = bgr.shape[:2]
        self.extractor.aspect = w / h
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.last_hand_result = self.hands.process(rgb, t) if self.hands else None
        self.last_face_result = self.face.process(rgb, t) if self.face else None
        return self.extractor.update(self.last_hand_result, self.last_face_result, t)

    def close(self) -> None:
        """Close both trackers (idempotent)."""
        for trk in (self.hands, self.face):
            if trk is not None:
                trk.close()

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
