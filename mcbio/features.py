"""Landmarks -> control features (head pose, blendshapes, palm position, finger curl, pinch).

This module is pure numpy: it consumes MediaPipe *result objects* (or plain
arrays in tests) and never imports mediapipe itself.

Coordinate frames used here
---------------------------
* **Image (normalized)**: MediaPipe landmarks on the *mirrored* frame. x in [0,1]
  grows to image-right (= the user's right, because the frame is mirrored),
  y in [0,1] grows downward, z is a relative depth (smaller = closer to the
  camera) on roughly the same scale as x. x is normalized by image width and y by
  image height, so distances need the aspect ratio ``aspect = width/height`` to be
  isotropic; isotropic distances in this module are in units of image height.
* **Camera (metric-like)**: right-handed, X = image-right, Y = up, Z = toward the
  viewer (the camera looks along -Z). This is MediaPipe's face-geometry "metric
  space", the frame of the facial transformation matrix.

Head-pose sign conventions (types.py)
-------------------------------------
yaw + = user turns to THEIR right, pitch + = looking up, roll + = head tilts
toward the user's right shoulder. Because the frame is mirrored, the user's right
is +X in the camera frame, so all three can be read directly off the face's
rotation in that frame (derivation in :func:`head_pose_from_rotation`).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from .filters import OneEuroFilter
from .types import Calibration, FaceFeatures, Features, HandFeatures, HeadPose

# --------------------------------------------------------------------------- #
# Hand model
# --------------------------------------------------------------------------- #
FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")

# MediaPipe 21-point hand topology. Each chain starts at the wrist (0) so the
# first bend angle is the MCP (thumb: CMC) flexion relative to the metacarpal.
FINGER_CHAINS: dict[str, tuple[int, int, int, int, int]] = {
    "thumb": (0, 1, 2, 3, 4),       # wrist, CMC, MCP, IP, tip
    "index": (0, 5, 6, 7, 8),       # wrist, MCP, PIP, DIP, tip
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "little": (0, 17, 18, 19, 20),
}

# Bend angle (deg) treated as "fully flexed" for each of the three joints in a
# chain -- roughly the active range of motion of the joint (MCP ~90, PIP ~100,
# DIP ~80; thumb CMC/MCP/IP are smaller). Curl = mean of the per-joint fractions,
# mirroring the "generalized flexion = mean of the 3 flexion joints" idea from
# the OpenSim pipeline, but normalized to [0, 1] instead of degrees.
JOINT_FULL_FLEX_DEG: dict[str, tuple[float, float, float]] = {
    "thumb": (45.0, 60.0, 80.0),
    "index": (90.0, 100.0, 80.0),
    "middle": (90.0, 100.0, 80.0),
    "ring": (90.0, 100.0, 80.0),
    "little": (90.0, 100.0, 80.0),
}

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP)

# --------------------------------------------------------------------------- #
# Face model (MediaPipe 478-point face mesh indices)
# --------------------------------------------------------------------------- #
# Left/right below are IMAGE sides of the frame that was fed to MediaPipe.
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
FACE_EDGE_IMG_LEFT = 234
FACE_EDGE_IMG_RIGHT = 454
EYE_OUTER_IMG_LEFT = 33
EYE_OUTER_IMG_RIGHT = 263

# Blendshape categories emitted by face_landmarker.task (ARKit-style names).
# flatten() always emits all of them so downstream bindings never see a missing key.
BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight",
)


# --------------------------------------------------------------------------- #
# Small geometry helpers
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _bend_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle (deg) between segments a->b and b->c: 0 when collinear (joint straight)."""
    u, v = b - a, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    # atan2(|u x v|, u . v) is numerically stable near 0 deg, unlike acos.
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v))))


def _image_to_isotropic(lms: np.ndarray, aspect: float) -> np.ndarray:
    """Normalized image (x,y,z) -> isotropic units of image height (x and z scaled by aspect)."""
    out = np.array(lms, dtype=float, copy=True)
    out[:, 0] *= aspect
    if out.shape[1] > 2:
        out[:, 2] *= aspect
    return out


# --------------------------------------------------------------------------- #
# Hand features
# --------------------------------------------------------------------------- #
def finger_curls(xyz: np.ndarray) -> tuple[float, float, float, float, float]:
    """Per-finger curl in [0,1] (thumb..little) from 21 isotropic 3-D landmarks.

    Each finger is a kinematic chain wrist -> MCP -> PIP -> DIP -> tip. At every
    interior joint the *bend angle* is the angle between the incoming and outgoing
    bone vectors (0 deg = straight). Angles between vectors are invariant to hand
    size, position and any rigid rotation, so the result does not depend on where
    or how the hand is held. Using z (MediaPipe's depth estimate) matters when the
    palm faces the camera: a curling finger then folds mostly toward the lens.
    """
    p = np.asarray(xyz, dtype=float)
    curls = []
    for name in FINGERS:
        chain = FINGER_CHAINS[name]
        full = JOINT_FULL_FLEX_DEG[name]
        fracs = [
            min(_bend_angle_deg(p[chain[j - 1]], p[chain[j]], p[chain[j + 1]]) / full[j - 1], 1.0)
            for j in (1, 2, 3)
        ]
        curls.append(float(np.mean(fracs)))
    return tuple(curls)  # type: ignore[return-value]


def hand_features_from_landmarks(
    lms21_xyz: Sequence[Sequence[float]] | np.ndarray,
    handedness: str,
    world_xyz: Sequence[Sequence[float]] | np.ndarray | None = None,
    aspect: float = 1.0,
) -> HandFeatures:
    """Build :class:`HandFeatures` from one hand's 21 landmarks.

    Parameters
    ----------
    lms21_xyz : (21, 3) normalized image landmarks of the mirrored frame.
    handedness : "Left"/"Right" as the user sees it (already mirror-corrected).
    world_xyz : optional (21, 3) MediaPipe *world* landmarks (metres, hand-centred).
        When given, curl and pinch are computed from them (true 3-D, no
        perspective/aspect distortion); otherwise from the image landmarks.
    aspect : image width / height, to make image distances isotropic.

    * palm  = mean of wrist, index MCP, middle MCP, pinky MCP (image coords).
      These four points sit on the rigid metacarpal "palm plate", so the centre
      is unaffected by finger motion.
    * scale = |wrist -> middle MCP| in image-height units (a palm-length proxy that
      shrinks as the hand moves away from the camera).
    * pinch = |thumb tip - index tip| / palm length (dimensionless).
    """
    img = np.asarray(lms21_xyz, dtype=float)
    if img.shape[0] != 21:
        raise ValueError(f"expected 21 hand landmarks, got {img.shape[0]}")
    if img.shape[1] == 2:
        img = np.column_stack([img, np.zeros(21)])

    iso = _image_to_isotropic(img, aspect)
    palm_xy = img[list(PALM_POINTS), :2].mean(axis=0)
    scale = float(np.linalg.norm(iso[MIDDLE_MCP, :2] - iso[WRIST, :2]))

    geom = np.asarray(world_xyz, dtype=float) if world_xyz is not None else iso
    palm_len = float(np.linalg.norm(geom[MIDDLE_MCP] - geom[WRIST]))
    pinch = float(np.linalg.norm(geom[THUMB_TIP] - geom[INDEX_TIP]) / palm_len) if palm_len > 1e-9 else 1.0

    return HandFeatures(
        present=True,
        handedness=handedness,
        palm=(float(palm_xy[0]), float(palm_xy[1])),
        curl=finger_curls(geom),
        pinch=pinch,
        scale=scale,
        landmarks=[tuple(map(float, pt)) for pt in img],
    )


def _swap_lr(label: str) -> str:
    return {"Left": "Right", "Right": "Left"}.get(label, label)


def hand_features_from_result(
    hand_result: Any, aspect: float = 1.0, swap_handedness: bool = False
) -> list[HandFeatures]:
    """Convert a MediaPipe ``HandLandmarkerResult`` to a list of HandFeatures.

    Handedness: MediaPipe's hand model labels handedness *assuming the input image
    is mirrored* (selfie view). Our frames ARE mirrored before tracking, so the
    label already names the user's real hand and ``swap_handedness`` should be
    False. Set it True only if un-mirrored frames are fed to the tracker.
    """
    if hand_result is None or not getattr(hand_result, "hand_landmarks", None):
        return []
    worlds = getattr(hand_result, "hand_world_landmarks", None) or []
    labels = getattr(hand_result, "handedness", None) or []
    hands = []
    for i, lms in enumerate(hand_result.hand_landmarks):
        label = labels[i][0].category_name if i < len(labels) and labels[i] else "Right"
        if swap_handedness:
            label = _swap_lr(label)
        img = [(lm.x, lm.y, lm.z) for lm in lms]
        world = [(lm.x, lm.y, lm.z) for lm in worlds[i]] if i < len(worlds) and worlds[i] else None
        hands.append(hand_features_from_landmarks(img, label, world, aspect))
    return hands


# --------------------------------------------------------------------------- #
# Head pose
# --------------------------------------------------------------------------- #
def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rotation_from_head_pose(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """3x3 face->camera rotation for a head pose in degrees (inverse of
    :func:`head_pose_from_rotation`). Useful for synthetic data and tests."""
    return _rot_y(math.radians(yaw)) @ _rot_x(-math.radians(pitch)) @ _rot_z(-math.radians(roll))


def head_pose_from_rotation(R: np.ndarray) -> HeadPose:
    """Head pose (deg) from a face->camera rotation in the camera frame (X right, Y up, Z to viewer).

    Sign derivation
    ---------------
    At R = I the face looks straight at the camera: its forward (out-of-nose)
    axis is +Z, its up axis +Y, and its left-to-right axis +X. In the mirrored
    frame +X is the *user's* right. We decompose R as the neck does it,
    intrinsic yaw -> pitch -> roll:

        R = Ry(a) @ Rx(b) @ Rz(c)

    * forward f = R e_z = (cos b sin a, -sin b, cos b cos a).
      Turning to the user's right swings the nose toward +X  => yaw = a = atan2(f_x, f_z).
      Looking up swings the nose toward +Y (f_y = -sin b)    => pitch = -b = asin(f_y).
    * row 1 of R = (cos b sin c, cos b cos c, -sin b), so c = atan2(R10, R11).
      A positive Rz(c) is counter-clockwise as seen from the camera, moving the
      top of the head toward -X (user's left). Tilting toward the user's RIGHT
      shoulder therefore means c < 0                         => roll = -c.
    """
    R = np.asarray(R, dtype=float)[:3, :3]
    yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    pitch = math.degrees(math.asin(float(np.clip(R[1, 2], -1.0, 1.0))))
    roll = math.degrees(math.atan2(-R[1, 0], R[1, 1]))
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


def head_pose_from_matrix(matrix: np.ndarray) -> HeadPose:
    """Head pose from MediaPipe's 4x4 facial transformation matrix.

    The matrix maps the canonical face model (metric, nose toward +Z, Y up) into
    the camera metric space (X right, Y up, camera looking along -Z), so its upper
    3x3 block is exactly the face->camera rotation of :func:`head_pose_from_rotation`.
    Because tracking runs on the mirrored frame, camera +X is the user's right and
    the yaw/roll signs come out in the user's frame with no extra flip. The block
    is re-orthonormalized (nearest rotation via SVD) in case it carries scale.
    """
    M = np.asarray(matrix, dtype=float).reshape(4, 4)
    U, _, Vt = np.linalg.svd(M[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:      # guard against a reflection
        U[:, -1] *= -1
        R = U @ Vt
    return head_pose_from_rotation(R)


def head_pose_from_landmarks(face_lms: Sequence[Sequence[float]] | np.ndarray, aspect: float = 1.0) -> HeadPose:
    """Fallback head pose from face-mesh geometry when no transformation matrix is available.

    Builds a face frame from four landmarks in camera-like axes (X right, Y up,
    Z toward camera; MediaPipe z is "smaller = closer", hence Z = -z):
      * right axis  r = image-left face edge (234) -> image-right face edge (454)
      * up axis     u = chin (152) -> forehead (10)
      * forward     f = r x u  (points out of the face, toward the camera at rest)
    then orthonormalizes and reuses :func:`head_pose_from_rotation`. Depends on
    MediaPipe's z estimate, so it is noisier than the matrix, and the chin->forehead
    line is not exactly vertical (a constant pitch offset that calibration removes).
    """
    p = np.asarray(face_lms, dtype=float)
    P = _image_to_isotropic(p[:, :3] if p.shape[1] >= 3 else np.column_stack([p, np.zeros(len(p))]), aspect)
    P[:, 1] *= -1.0   # y down -> Y up
    P[:, 2] *= -1.0   # z (smaller = closer) -> Z toward camera
    r = P[FACE_EDGE_IMG_RIGHT] - P[FACE_EDGE_IMG_LEFT]
    u = P[FOREHEAD] - P[CHIN]
    f = _unit(np.cross(r, u))
    y = _unit(np.cross(f, r))
    x = np.cross(y, f)
    return head_pose_from_rotation(np.column_stack([x, y, f]))


# --------------------------------------------------------------------------- #
# Face features
# --------------------------------------------------------------------------- #
def face_features_from_result(face_result: Any, aspect: float = 1.0) -> FaceFeatures:
    """Convert a MediaPipe ``FaceLandmarkerResult`` (first face) to FaceFeatures.

    Head pose comes from the facial transformation matrix when present, otherwise
    from :func:`head_pose_from_landmarks`. ``nose`` is the nose tip in mirrored
    normalized image coords, ``scale`` the outer-eye-corner distance in
    image-height units. Blendshape names are passed through as MediaPipe reports
    them (note: their Left/Right suffixes are as the model interprets the
    mirrored image; verify on the HUD before binding a one-sided expression).
    """
    if face_result is None or not getattr(face_result, "face_landmarks", None):
        return FaceFeatures(present=False)
    lms = np.array([(lm.x, lm.y, lm.z) for lm in face_result.face_landmarks[0]], dtype=float)

    matrices = getattr(face_result, "facial_transformation_matrixes", None) or []
    head = head_pose_from_matrix(matrices[0]) if len(matrices) else head_pose_from_landmarks(lms, aspect)

    shapes = getattr(face_result, "face_blendshapes", None) or []
    blend = {c.category_name: float(c.score) for c in shapes[0]} if shapes else {}

    iso = _image_to_isotropic(lms, aspect)
    scale = float(np.linalg.norm(iso[EYE_OUTER_IMG_RIGHT, :2] - iso[EYE_OUTER_IMG_LEFT, :2]))
    return FaceFeatures(
        present=True,
        head=head,
        blend=blend,
        nose=(float(lms[NOSE_TIP, 0]), float(lms[NOSE_TIP, 1])),
        scale=scale,
        landmarks=lms.tolist(),
    )


def features_from_results(
    hand_result: Any, face_result: Any, t: float, aspect: float = 1.0, swap_handedness: bool = False
) -> Features:
    """Unsmoothed Features for one frame. Either result may be None (tracker disabled)."""
    return Features(
        t=float(t),
        face=face_features_from_result(face_result, aspect),
        hands=hand_features_from_result(hand_result, aspect, swap_handedness),
    )


# --------------------------------------------------------------------------- #
# Stateful smoothing
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """Per-frame feature extraction with One Euro smoothing.

    Smoothed: head pose (yaw, pitch, roll), and per hand palm position, curls and
    pinch. Hands are tracked across frames by their handedness label; a hand's
    filters are reset when it disappears so a re-appearing hand does not glide in
    from its old position. Blendshapes, nose and raw landmarks are not smoothed
    (gesture detection applies its own hysteresis).

    Each ``(min_cutoff, beta)`` pair is in the units of its signal: degrees for
    the head, normalized image units for the palm, unitless for curl and pinch.
    """

    def __init__(
        self,
        aspect: float = 1.0,
        swap_handedness: bool = False,
        head_params: tuple[float, float] = (1.0, 0.02),
        palm_params: tuple[float, float] = (1.5, 2.0),
        curl_params: tuple[float, float] = (2.0, 1.0),
        pinch_params: tuple[float, float] = (2.0, 1.0),
        d_cutoff: float = 1.0,
    ) -> None:
        self.aspect = float(aspect)
        self.swap_handedness = swap_handedness
        self._params = {"head": head_params, "palm": palm_params, "curl": curl_params, "pinch": pinch_params}
        self._d_cutoff = d_cutoff
        self._head = self._make("head")
        self._hand_filters: dict[str, dict[str, OneEuroFilter]] = {}

    def _make(self, kind: str) -> OneEuroFilter:
        mc, beta = self._params[kind]
        return OneEuroFilter(min_cutoff=mc, beta=beta, d_cutoff=self._d_cutoff)

    def reset(self) -> None:
        """Drop all filter history (e.g. after pause or recalibration)."""
        self._head.reset()
        self._hand_filters.clear()

    def update(self, hand_result: Any, face_result: Any, t: float) -> Features:
        """Extract and smooth features for the frame captured at monotonic time ``t``."""
        feats = features_from_results(hand_result, face_result, t, self.aspect, self.swap_handedness)
        self.smooth(feats)
        return feats

    def smooth(self, feats: Features) -> Features:
        """Smooth an already-extracted Features in place (also returned)."""
        t = feats.t
        if feats.face.present:
            h = feats.face.head
            y, p, r = self._head((h.yaw, h.pitch, h.roll), t)
            feats.face.head = HeadPose(float(y), float(p), float(r))
        else:
            self._head.reset()

        seen: set[str] = set()
        for hand in feats.hands:
            key = hand.handedness if hand.handedness not in seen else f"{hand.handedness}#2"
            seen.add(key)
            f = self._hand_filters.setdefault(key, {k: self._make(k) for k in ("palm", "curl", "pinch")})
            px, py = f["palm"](hand.palm, t)
            hand.palm = (float(px), float(py))
            hand.curl = tuple(float(np.clip(c, 0.0, 1.0)) for c in f["curl"](hand.curl, t))  # type: ignore[assignment]
            hand.pinch = float(max(f["pinch"](hand.pinch, t), 0.0))
        for key in list(self._hand_filters):
            if key not in seen:
                del self._hand_filters[key]
        return feats


# --------------------------------------------------------------------------- #
# Flatten for generic bindings
# --------------------------------------------------------------------------- #
def select_hands(hands: Iterable[HandFeatures], dominant: str = "Right") -> tuple[HandFeatures | None, HandFeatures | None]:
    """Return (dominant, offhand). Dominant = first hand labelled ``dominant``,
    else the first hand present; offhand = the other hand if there is one."""
    present = [h for h in hands if h.present]
    if not present:
        return None, None
    dom = next((h for h in present if h.handedness == dominant), present[0])
    rest = [h for h in present if h is not dom]
    return dom, (rest[0] if rest else None)


def flatten(features: Features, calib: Calibration, dominant: str | None = None) -> dict[str, float]:
    """Flatten Features into calibrated, signed scalars keyed by dotted source names (SPEC.md).

    ``dominant`` defaults to ``calib.dominant_hand`` ("Right" unless calibrated
    otherwise). Every key is always present; absent face/hand values are 0.
    Signs: yaw/pitch/roll as in types.py; ``palm_dx`` + = right, ``palm_dy`` + = UP
    (image y flipped), ``nose_dx`` + = right, ``nose_dy`` + = UP, ``scale_ratio``
    + = leaning toward the camera.
    """
    dominant = dominant or calib.dominant_hand
    out: dict[str, float] = {}

    # ---- face ----
    face = features.face
    blend_names = set(BLENDSHAPE_NAMES) | set(face.blend) | set(calib.neutral_blend)
    if face.present:
        out["face.present"] = 1.0
        out["face.yaw"] = face.head.yaw - calib.neutral_yaw
        out["face.pitch"] = face.head.pitch - calib.neutral_pitch
        out["face.roll"] = face.head.roll - calib.neutral_roll
        for name in blend_names:
            out[f"face.blend.{name}"] = max(face.blend.get(name, 0.0) - calib.neutral_blend.get(name, 0.0), 0.0)
        if face.scale > 0:
            out["face.nose_dx"] = (face.nose[0] - calib.neutral_nose[0]) / face.scale
            out["face.nose_dy"] = -(face.nose[1] - calib.neutral_nose[1]) / face.scale
        else:
            out["face.nose_dx"] = out["face.nose_dy"] = 0.0
        out["face.scale_ratio"] = face.scale / calib.neutral_face_scale - 1.0 if calib.neutral_face_scale > 0 else 0.0
    else:
        for k in ("face.present", "face.yaw", "face.pitch", "face.roll", "face.nose_dx", "face.nose_dy", "face.scale_ratio"):
            out[k] = 0.0
        for name in blend_names:
            out[f"face.blend.{name}"] = 0.0

    # ---- hands ----
    dom, off = select_hands(features.hands, dominant)
    rng = calib.palm_range if calib.palm_range > 0 else 1.0
    if dom is not None:
        out["hand.present"] = 1.0
        out["hand.palm_dx"] = (dom.palm[0] - calib.neutral_palm[0]) / rng
        out["hand.palm_dy"] = -(dom.palm[1] - calib.neutral_palm[1]) / rng
        out["hand.curl_mean"] = float(np.mean(dom.curl[1:]))
        for name, c in zip(FINGERS, dom.curl):
            out[f"hand.curl_{name}"] = float(c)
        out["hand.pinch"] = float(dom.pinch)
    else:
        for k in ("hand.present", "hand.palm_dx", "hand.palm_dy", "hand.curl_mean", "hand.pinch"):
            out[k] = 0.0
        for name in FINGERS:
            out[f"hand.curl_{name}"] = 0.0

    if off is not None:
        out["offhand.present"] = 1.0
        out["offhand.curl_mean"] = float(np.mean(off.curl[1:]))
        out["offhand.pinch"] = float(off.pinch)
    else:
        out["offhand.present"] = out["offhand.curl_mean"] = out["offhand.pinch"] = 0.0

    out["hands.count"] = float(sum(1 for h in features.hands if h.present))
    return out
