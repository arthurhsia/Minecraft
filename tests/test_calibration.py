"""Tests for mcbio.calibration."""
import random

import pytest

from mcbio import calibration as cal
from mcbio.types import Calibration, FaceFeatures, Features, HandFeatures, HeadPose


def frame(t, yaw=0.0, pitch=0.0, roll=0.0, palm=(0.5, 0.6), face=True, hand=True,
          handedness="Right", jaw=0.05):
    hands = [HandFeatures(present=True, handedness=handedness, palm=palm)] if hand else []
    return Features(
        t=t,
        face=FaceFeatures(present=face, head=HeadPose(yaw, pitch, roll),
                          blend={"jawOpen": jaw, "browInnerUp": 0.1},
                          nose=(0.48, 0.52), scale=0.1),
        hands=hands,
    )


def test_medians_are_robust_to_outliers():
    rng = random.Random(0)
    samples = [frame(i / 30, yaw=5 + rng.gauss(0, 0.5), pitch=-3 + rng.gauss(0, 0.5),
                     roll=2.0, palm=(0.55 + rng.gauss(0, 0.002), 0.65)) for i in range(60)]
    # A few wild tracking glitches.
    samples += [frame(2.0 + i / 30, yaw=80, pitch=60, palm=(0.0, 0.0), jaw=1.0) for i in range(5)]
    c = cal.calibrate(samples, dominant_hand="Right")
    assert c.neutral_yaw == pytest.approx(5, abs=0.5)
    assert c.neutral_pitch == pytest.approx(-3, abs=0.5)
    assert c.neutral_roll == pytest.approx(2.0)
    assert c.neutral_palm == pytest.approx((0.55, 0.65), abs=0.01)
    assert c.neutral_nose == pytest.approx((0.48, 0.52))
    assert c.neutral_face_scale == pytest.approx(0.1)
    assert c.neutral_blend["jawOpen"] == pytest.approx(0.05)
    assert c.palm_range == cal.DEFAULT_PALM_RANGE


def test_dominant_hand_selection():
    s = [Features(t=0, hands=[HandFeatures(present=True, handedness="Right", palm=(0.7, 0.5)),
                              HandFeatures(present=True, handedness="Left", palm=(0.2, 0.5))])]
    assert cal.calibrate(s, dominant_hand="Left").neutral_palm == (0.2, 0.5)
    assert cal.calibrate(s, dominant_hand="Right").neutral_palm == (0.7, 0.5)


def test_face_only_and_hand_only_calibrate():
    c = cal.calibrate([frame(i, yaw=4, hand=False) for i in range(5)])
    assert c.neutral_yaw == 4 and c.neutral_palm == Calibration().neutral_palm
    c = cal.calibrate([frame(i, face=False, palm=(0.4, 0.4)) for i in range(5)])
    assert c.neutral_palm == (0.4, 0.4) and c.neutral_yaw == 0.0
    with pytest.raises(cal.CalibrationError):
        cal.calibrate([frame(0, face=False, hand=False)])


def test_calibrator_window():
    c = cal.Calibrator(duration_s=2.0)
    done = [c.add(frame(i / 30, yaw=3)) for i in range(70)]
    assert done.index(True) == 60
    assert c.result().neutral_yaw == 3
    with pytest.raises(cal.CalibrationError):
        cal.Calibrator(min_samples=10).result()


def test_capture_range():
    base = Calibration(neutral_palm=(0.5, 0.5))
    reach = [frame(i, palm=(0.5 + 0.2 * (i % 2 * 2 - 1), 0.5)) for i in range(20)]
    reach.append(frame(99, palm=(1.0, 1.0)))     # one wild frame
    c = cal.capture_range(reach, base)
    assert c.palm_range == pytest.approx(0.2)
    assert base.palm_range == 0.12               # original untouched
    assert cal.capture_range([frame(0, hand=False)], base).palm_range == 0.12
    tiny = cal.capture_range([frame(0, palm=(0.5, 0.51))], base)
    assert tiny.palm_range == cal.MIN_PALM_RANGE


def test_json_round_trip(tmp_path):
    c = Calibration(neutral_yaw=1.5, neutral_pitch=-2.0, neutral_roll=0.5,
                    neutral_nose=(0.4, 0.6), neutral_face_scale=0.11,
                    neutral_palm=(0.55, 0.62), palm_range=0.15,
                    neutral_blend={"jawOpen": 0.03}, dominant_hand="Left")
    path = cal.save(c, tmp_path / "sub" / "calib.json")
    assert path.exists()
    assert cal.load(path) == c
    with pytest.raises(FileNotFoundError):
        cal.load(tmp_path / "missing.json")


def test_default_path():
    assert str(cal.DEFAULT_PATH).endswith(".mcbio/calibration.json")
