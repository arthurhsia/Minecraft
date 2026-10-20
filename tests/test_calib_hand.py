"""Calibration must take the joystick centre from where the hand really rests."""
from mcbio.app import CalibrationFlow
from mcbio.types import Features, FaceFeatures, HeadPose, HandFeatures


def _feats(t, hand_at=None):
    face = FaceFeatures(True, HeadPose(0, 0, 0), {}, (0.5, 0.4), 0.1)
    hands = []
    if hand_at:
        hands = [HandFeatures(True, "Right", hand_at, (0,) * 5, 1.0, 0.1)]
    return Features(t, face, hands)


def _run(flow, gen, t0=0.0, seconds=20.0, dt=1 / 30):
    t = t0
    while t < t0 + seconds and not flow.done:
        flow.update(gen(t), t)
        t += dt
    return t


def test_neutral_follows_hand_off_to_the_side():
    flow = CalibrationFlow(require_hand=True)
    _run(flow, lambda t: _feats(t, (0.8, 0.45)))
    assert flow.done
    assert abs(flow.result.neutral_palm[0] - 0.8) < 1e-6


def test_retries_when_hand_not_visible_then_uses_it():
    flow = CalibrationFlow(require_hand=True)
    t = _run(flow, lambda t: _feats(t), seconds=5.5)   # first attempt: no hand
    assert not flow.done and flow.retries == 1
    _run(flow, lambda t: _feats(t, (0.78, 0.5)), t0=t)
    assert flow.done and abs(flow.result.neutral_palm[0] - 0.78) < 1e-6


def test_face_only_profile_does_not_wait_for_hand():
    flow = CalibrationFlow(require_hand=False)
    _run(flow, lambda t: _feats(t))
    assert flow.done and flow.retries == 0
