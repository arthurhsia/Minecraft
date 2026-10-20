"""Synthetic tests for mcbio.features (no mediapipe models, no camera)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from mcbio import features as F
from mcbio.types import Calibration, FaceFeatures, Features, HandFeatures, HeadPose


# --------------------------------------------------------------------------- #
# Synthetic hand: palm in the x-y plane, fingers pointing toward -y (image "up").
# Each finger is a 4-bone chain whose joints bend by the given angles about the
# flexion axis (perpendicular to the finger, in the palm plane).
# --------------------------------------------------------------------------- #
MCP_X = {"index": -0.3, "middle": -0.1, "ring": 0.1, "little": 0.3}
BONES = (0.45, 0.25, 0.18, 0.15)   # metacarpal-ish (wrist->MCP), proximal, middle, distal


def _chain(base, direction, lengths, bends_deg, axis):
    """Points after `base` following `direction`, rotating by each bend about `axis`."""
    pts, p, d = [], np.asarray(base, float), np.asarray(direction, float)
    for length, bend in zip(lengths, bends_deg):
        a = math.radians(bend)
        k = axis / np.linalg.norm(axis)
        # Rodrigues rotation of d about k
        d = d * math.cos(a) + np.cross(k, d) * math.sin(a) + k * np.dot(k, d) * (1 - math.cos(a))
        p = p + length * d
        pts.append(p)
    return pts


def synthetic_hand(finger_bends=(0, 0, 0), thumb_bends=(0, 0, 0)):
    """21x3 landmarks. finger_bends = (MCP, PIP, DIP) applied to all four fingers."""
    lms = np.zeros((21, 3))
    wrist = np.zeros(3)
    lms[0] = wrist
    for name, x in MCP_X.items():
        chain = F.FINGER_CHAINS[name]
        mcp = np.array([x, -BONES[0], 0.0])
        lms[chain[1]] = mcp
        d0 = (mcp - wrist) / np.linalg.norm(mcp - wrist)
        axis = np.cross(d0, [0, 0, 1.0])     # flexion axis perpendicular to the finger, in the palm plane
        for idx, p in zip(chain[2:], _chain(mcp, d0, BONES[1:], finger_bends, axis)):
            lms[idx] = p
    # Thumb: CMC off to the side, pointing up-left; bends about an axis in the palm plane.
    cmc = np.array([-0.2, -0.1, 0.0])
    lms[1] = cmc
    d0 = cmc / np.linalg.norm(cmc)
    axis = np.cross(d0, [0, 0, 1.0])
    for idx, p in zip((2, 3, 4), _chain(cmc, d0, (0.2, 0.15, 0.12), thumb_bends, axis)):
        lms[idx] = p
    return lms


def _place(lms, scale=0.3, angle_deg=0.0, center=(0.5, 0.5)):
    """Scale + in-plane rotate + translate into normalized image coordinates."""
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    out = (lms @ R.T) * scale
    out[:, 0] += center[0]
    out[:, 1] += center[1]
    return out


FIST = dict(finger_bends=(90, 100, 75), thumb_bends=(30, 50, 60))


def test_straight_hand_curl_near_zero():
    hf = F.hand_features_from_landmarks(_place(synthetic_hand()), "Right")
    assert hf.present and hf.handedness == "Right"
    assert max(hf.curl) < 0.05


def test_fist_curl_high():
    hf = F.hand_features_from_landmarks(_place(synthetic_hand(**FIST)), "Right")
    assert min(hf.curl[1:]) > 0.85
    assert hf.curl[0] > 0.6


def test_partial_curl_is_monotonic():
    curls = [np.mean(F.hand_features_from_landmarks(_place(synthetic_hand((b, b, b))), "Right").curl[1:])
             for b in (0, 20, 45, 70)]
    assert all(a < b for a, b in zip(curls, curls[1:]))


@pytest.mark.parametrize("scale,angle", [(0.1, 0), (0.5, 0), (0.3, 37), (0.2, -120), (0.4, 180)])
def test_curl_scale_and_rotation_invariant(scale, angle):
    ref = F.hand_features_from_landmarks(_place(synthetic_hand((40, 60, 30))), "Right").curl
    got = F.hand_features_from_landmarks(_place(synthetic_hand((40, 60, 30)), scale, angle), "Right").curl
    np.testing.assert_allclose(got, ref, atol=1e-7)


def test_palm_scale_pinch():
    lms = _place(synthetic_hand(), scale=0.3, center=(0.4, 0.6))
    hf = F.hand_features_from_landmarks(lms, "Left")
    expected_palm = lms[[0, 5, 9, 17], :2].mean(axis=0)
    assert hf.palm == pytest.approx(tuple(expected_palm))
    assert hf.scale == pytest.approx(0.3 * BONES[0] * math.hypot(1, 0.1 / BONES[0]), rel=1e-6)
    assert hf.pinch > 0.5                     # open hand: tips apart
    assert len(hf.landmarks) == 21


def test_pinch_small_when_tips_touch():
    lms = _place(synthetic_hand())
    lms[4] = lms[8] + np.array([0.003, 0.0, 0.0])
    hf = F.hand_features_from_landmarks(lms, "Right")
    assert hf.pinch < 0.05


def test_world_landmarks_used_for_curl():
    straight_img = _place(synthetic_hand())
    fist_world = synthetic_hand(**FIST) * 0.09
    hf = F.hand_features_from_landmarks(straight_img, "Right", world_xyz=fist_world)
    assert min(hf.curl[1:]) > 0.85


# --------------------------------------------------------------------------- #
# Head pose
# --------------------------------------------------------------------------- #
POSES = [(0, 0, 0), (20, 0, 0), (-25, 0, 0), (0, 15, 0), (0, -10, 0), (0, 0, 12), (0, 0, -18),
         (15, -10, 8), (-30, 20, -12)]


def _matrix(yaw, pitch, roll):
    M = np.eye(4)
    M[:3, :3] = F.rotation_from_head_pose(yaw, pitch, roll)
    M[:3, 3] = (1.0, -2.0, -50.0)
    return M


@pytest.mark.parametrize("yaw,pitch,roll", POSES)
def test_head_pose_from_matrix_roundtrip(yaw, pitch, roll):
    hp = F.head_pose_from_matrix(_matrix(yaw, pitch, roll))
    assert (hp.yaw, hp.pitch, hp.roll) == pytest.approx((yaw, pitch, roll), abs=1e-6)


def test_head_pose_signs_physical():
    """Check signs against the geometry, independent of rotation_from_head_pose."""
    # Turn to user's right = nose (face +Z) swings toward camera +X (user's right in mirrored view).
    a = math.radians(20)
    turn_right = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    assert turn_right @ [0, 0, 1] == pytest.approx([math.sin(a), 0, math.cos(a)])
    assert F.head_pose_from_rotation(turn_right).yaw == pytest.approx(20)
    # Look up = nose swings toward +Y.
    look_up = np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])
    assert (look_up @ [0, 0, 1])[1] > 0
    assert F.head_pose_from_rotation(look_up).pitch == pytest.approx(20)
    # Tilt toward right shoulder = top of head (face +Y) swings toward +X.
    tilt_right = np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]])
    assert (tilt_right @ [0, 1, 0])[0] > 0
    assert F.head_pose_from_rotation(tilt_right).roll == pytest.approx(20)


def test_head_pose_from_matrix_tolerates_scale():
    M = _matrix(10, -5, 7)
    M[:3, :3] *= 1.3
    hp = F.head_pose_from_matrix(M)
    assert (hp.yaw, hp.pitch, hp.roll) == pytest.approx((10, -5, 7), abs=1e-6)


# Canonical face points (camera-like axes: X right, Y up, Z toward camera).
FACE_PTS = {F.FACE_EDGE_IMG_LEFT: (-1.0, 0.0, 0.0), F.FACE_EDGE_IMG_RIGHT: (1.0, 0.0, 0.0),
            F.FOREHEAD: (0.0, 1.2, 0.0), F.CHIN: (0.0, -1.2, 0.0), F.NOSE_TIP: (0.0, -0.1, 0.8),
            F.EYE_OUTER_IMG_LEFT: (-0.6, 0.35, 0.2), F.EYE_OUTER_IMG_RIGHT: (0.6, 0.35, 0.2)}


def synthetic_face(yaw, pitch, roll, aspect=16 / 9, size=0.08, center=(0.5, 0.45)):
    """478 normalized face landmarks (only the indices above are meaningful)."""
    R = F.rotation_from_head_pose(yaw, pitch, roll)
    lms = np.zeros((478, 3))
    for idx, p in FACE_PTS.items():
        X, Y, Z = (R @ np.asarray(p)) * size
        lms[idx] = (center[0] + X / aspect, center[1] - Y, -Z / aspect)
    return lms


@pytest.mark.parametrize("yaw,pitch,roll", POSES)
def test_head_pose_from_landmarks(yaw, pitch, roll):
    hp = F.head_pose_from_landmarks(synthetic_face(yaw, pitch, roll), aspect=16 / 9)
    assert (hp.yaw, hp.pitch, hp.roll) == pytest.approx((yaw, pitch, roll), abs=1e-6)


def _lm(arr):
    return [SimpleNamespace(x=float(a[0]), y=float(a[1]), z=float(a[2])) for a in arr]


def test_face_features_from_result():
    lms = synthetic_face(12, -4, 3)
    res = SimpleNamespace(
        face_landmarks=[_lm(lms)],
        face_blendshapes=[[SimpleNamespace(category_name="jawOpen", score=0.7)]],
        facial_transformation_matrixes=[_matrix(12, -4, 3)],
    )
    ff = F.face_features_from_result(res, aspect=16 / 9)
    assert ff.present and ff.blend == {"jawOpen": 0.7}
    assert ff.head.yaw == pytest.approx(12) and ff.head.pitch == pytest.approx(-4)
    assert ff.nose == pytest.approx(tuple(lms[F.NOSE_TIP, :2]))
    assert ff.scale > 0
    # Without the matrix the landmark fallback is used.
    res.facial_transformation_matrixes = []
    assert F.face_features_from_result(res, aspect=16 / 9).head.yaw == pytest.approx(12, abs=1e-6)
    assert not F.face_features_from_result(None).present


def _hand_result(hands):
    """Fake HandLandmarkerResult from [(label, 21x3 array)]."""
    return SimpleNamespace(
        hand_landmarks=[_lm(a) for _, a in hands],
        hand_world_landmarks=[],
        handedness=[[SimpleNamespace(category_name=lbl, score=0.9)] for lbl, _ in hands],
    )


def test_hand_result_handedness_swap():
    res = _hand_result([("Left", _place(synthetic_hand()))])
    assert F.hand_features_from_result(res)[0].handedness == "Left"
    assert F.hand_features_from_result(res, swap_handedness=True)[0].handedness == "Right"
    assert F.hand_features_from_result(None) == []


def test_feature_extractor_smooths_and_resets():
    ex = F.FeatureExtractor()
    still = _place(synthetic_hand(), center=(0.5, 0.5))
    moved = _place(synthetic_hand(), center=(0.7, 0.5))
    f0 = ex.update(_hand_result([("Right", still)]), None, 0.0)
    f1 = ex.update(_hand_result([("Right", moved)]), None, 1 / 30)
    raw_x = moved[[0, 5, 9, 17], 0].mean()
    assert f0.hands[0].palm[0] < f1.hands[0].palm[0] < raw_x      # lags behind the jump
    assert not f1.face.present
    ex.update(_hand_result([]), None, 2 / 30)                       # hand lost -> filters dropped
    f3 = ex.update(_hand_result([("Right", moved)]), None, 3 / 30)
    assert f3.hands[0].palm[0] == pytest.approx(raw_x)              # re-acquired: no glide


# --------------------------------------------------------------------------- #
# flatten
# --------------------------------------------------------------------------- #
HAND_KEYS = {"hand.present", "hand.palm_dx", "hand.palm_dy", "hand.curl_mean", "hand.pinch",
             *(f"hand.curl_{n}" for n in F.FINGERS)}
OTHER_KEYS = {"face.present", "face.yaw", "face.pitch", "face.roll", "face.nose_dx", "face.nose_dy",
              "face.scale_ratio", "offhand.present", "offhand.curl_mean", "offhand.pinch", "hands.count"}


def test_flatten_empty_all_zero():
    out = F.flatten(Features(t=0.0), Calibration())
    expected = HAND_KEYS | OTHER_KEYS | {f"face.blend.{n}" for n in F.BLENDSHAPE_NAMES}
    assert expected <= set(out)
    assert all(v == 0.0 for v in out.values())


def test_flatten_signs_and_selection():
    calib = Calibration(neutral_yaw=2.0, neutral_pitch=-1.0, neutral_roll=0.5, neutral_nose=(0.5, 0.4),
                        neutral_face_scale=0.10, neutral_palm=(0.5, 0.6), palm_range=0.1,
                        neutral_blend={"jawOpen": 0.1})
    face = FaceFeatures(present=True, head=HeadPose(12.0, 4.0, -3.5), blend={"jawOpen": 0.6, "browInnerUp": 0.05},
                        nose=(0.55, 0.38), scale=0.11)
    right = HandFeatures(present=True, handedness="Right", palm=(0.6, 0.5), curl=(0.1, 0.8, 0.8, 0.6, 0.6),
                         pinch=0.2, scale=0.1)
    left = HandFeatures(present=True, handedness="Left", palm=(0.2, 0.5), curl=(0, 0, 0, 0, 0), pinch=1.2, scale=0.1)
    out = F.flatten(Features(t=1.0, face=face, hands=[left, right]), calib)

    assert out["face.present"] == 1.0
    assert out["face.yaw"] == pytest.approx(10.0)
    assert out["face.pitch"] == pytest.approx(5.0)
    assert out["face.roll"] == pytest.approx(-4.0)
    assert out["face.blend.jawOpen"] == pytest.approx(0.5)
    assert out["face.blend.browInnerUp"] == pytest.approx(0.05)
    assert out["face.blend.mouthPucker"] == 0.0
    assert out["face.nose_dx"] == pytest.approx(0.05 / 0.11)      # + = right
    assert out["face.nose_dy"] == pytest.approx(0.02 / 0.11)      # nose moved up -> +
    assert out["face.scale_ratio"] == pytest.approx(0.1)          # leaning in -> +

    assert out["hand.present"] == 1.0                             # Right is dominant
    assert out["hand.palm_dx"] == pytest.approx(1.0)              # moved right
    assert out["hand.palm_dy"] == pytest.approx(1.0)              # y smaller = UP -> +
    assert out["hand.curl_mean"] == pytest.approx(0.7)
    assert out["hand.curl_thumb"] == pytest.approx(0.1)
    assert out["hand.pinch"] == pytest.approx(0.2)
    assert out["offhand.present"] == 1.0
    assert out["offhand.pinch"] == pytest.approx(1.2)
    assert out["hands.count"] == 2.0

    # Left-dominant calibration swaps roles; blend below neutral clips to 0.
    face.blend["jawOpen"] = 0.05
    out_l = F.flatten(Features(t=1.0, face=face, hands=[left, right]), calib, dominant="Left")
    assert out_l["hand.palm_dx"] == pytest.approx(-3.0) and out_l["offhand.pinch"] == pytest.approx(0.2)
    assert out_l["face.blend.jawOpen"] == 0.0


def test_flatten_single_hand_fallback_is_dominant():
    left = HandFeatures(present=True, handedness="Left", palm=(0.5, 0.7), curl=(0, 1, 1, 1, 1), pinch=0.5, scale=0.1)
    out = F.flatten(Features(t=0.0, hands=[left]), Calibration(dominant_hand="Right"))
    assert out["hand.present"] == 1.0 and out["hand.curl_mean"] == pytest.approx(1.0)
    assert out["hand.palm_dy"] == pytest.approx(-0.1 / 0.12)      # moved down -> -
    assert out["offhand.present"] == 0.0 and out["hands.count"] == 1.0
    assert out["face.present"] == 0.0 and out["face.yaw"] == 0.0
