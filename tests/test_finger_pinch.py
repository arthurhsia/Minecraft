"""Thumb-to-finger pinches (middle / ring / little) -> one-shot taps, and the simple HUD."""
from pathlib import Path

import numpy as np
import pytest

from mcbio import features as F
from mcbio.mapping import Mapper, Profile
from mcbio.outputs import SerialLeonardoOutput
from mcbio.types import Calibration, Features

PROFILES = Path(__file__).resolve().parent.parent / "profiles"
DT = 1 / 30.0


def flat(**over) -> dict[str, float]:
    """Neutral flat dict (hand present, open, no pinch) with overrides; keys use '__' for '.'."""
    d = {"face.present": 1.0, "face.yaw": 0.0, "face.pitch": 0.0, "face.roll": 0.0,
         "hand.present": 1.0, "hand.palm_dx": 0.0, "hand.palm_dy": 0.0, "hand.curl_mean": 0.1,
         "hand.pinch": 1.0, "hand.pinch_middle": 1.1, "hand.pinch_ring": 1.2, "hand.pinch_little": 1.3,
         "offhand.present": 0.0, "hands.count": 1.0}
    d.update({k.replace("__", "."): v for k, v in over.items()})
    return d


def feed(mapper, values, seconds, t0=0.0):
    """Feed the same dict for `seconds`; return (all taps in order, final Commands, end time)."""
    taps, cmds, t = [], None, t0
    for _ in range(int(round(seconds / DT))):
        cmds = mapper.update(values, t)
        taps += cmds.taps
        t += DT
    return taps, cmds, t


@pytest.fixture
def mapper():
    return Mapper(Profile.load(PROFILES / "face_hand.yaml"))


# ------------------------------------------------------------------ features
def _hand_touching(tip_index: int) -> np.ndarray:
    """21 landmarks: wrist->middle MCP = 1 unit, fingertips fanned out, thumb tip on one tip."""
    lms = np.zeros((21, 3))
    lms[9] = (0.0, -1.0, 0.0)                                   # middle MCP -> palm length 1
    for mcp, x in ((5, -0.3), (13, 0.3), (17, 0.6)):
        lms[mcp] = (x, -0.95, 0.0)
    for tip, x in ((8, -0.4), (12, 0.0), (16, 0.4), (20, 0.8)):
        lms[tip] = (x, -2.0, 0.0)
        lms[tip - 1] = (x, -1.7, 0.0)
        lms[tip - 2] = (x, -1.4, 0.0)
    lms[1], lms[2], lms[3] = (-0.4, -0.2, 0), (-0.6, -0.6, 0), (-0.5, -1.2, 0)
    lms[4] = lms[tip_index] + (0.02, 0.0, 0.0)
    return lms


def test_finger_pinch_distances_normalised_by_palm():
    h = F.hand_features_from_landmarks(_hand_touching(12), "Right")
    middle, ring, little = h.finger_pinch
    assert middle == pytest.approx(0.02, abs=1e-6)
    assert ring > 0.3 and little > 0.7 and h.pinch > 0.3      # others clearly apart
    # Scale invariance: the same hand 3x bigger gives the same ratios.
    h2 = F.hand_features_from_landmarks(_hand_touching(12) * 3.0, "Right")
    assert np.allclose(h.finger_pinch, h2.finger_pinch)


def test_flatten_exposes_finger_pinches():
    h = F.hand_features_from_landmarks(_hand_touching(16), "Right")
    out = F.flatten(Features(t=0.0, hands=[h]), Calibration())
    assert out["hand.pinch_ring"] == pytest.approx(0.02, abs=1e-6)
    assert {"hand.pinch_middle", "hand.pinch_little"} <= set(out)
    empty = F.flatten(Features(t=0.0), Calibration())
    assert empty["hand.pinch_middle"] == empty["hand.pinch_ring"] == empty["hand.pinch_little"] == 0.0


def test_extractor_smooths_finger_pinch():
    ex = F.FeatureExtractor()
    for i in range(5):
        h = F.hand_features_from_landmarks(_hand_touching(20), "Right")
        f = ex.smooth(Features(t=i * DT, hands=[h]))
    assert len(f.hands[0].finger_pinch) == 3 and f.hands[0].finger_pinch[2] < 0.1


# ------------------------------------------------------------------ mapping / taps
def test_middle_pinch_taps_inventory_once(mapper):
    taps, cmds, t = feed(mapper, flat(), 0.3)
    assert taps == []
    # Thumb on the middle tip; the index tip sits right next to it (0.25 < pinch "on").
    taps, cmds, t = feed(mapper, flat(hand__pinch_middle=0.05, hand__pinch=0.25), 1.5, t)
    assert taps == ["inventory"]                       # once, no repeat
    assert not cmds.right_button                       # nearest finger wins: no index pinch
    taps, _, t = feed(mapper, flat(), 0.3, t)          # release
    taps, _, t = feed(mapper, flat(hand__pinch_middle=0.05), 0.3, t)
    assert taps == ["inventory"]                       # a new pinch taps again


def test_ring_and_little_pinch_step_hotbar_with_repeat(mapper):
    taps, _, t = feed(mapper, flat(hand__pinch_ring=0.05), 1.0)
    # fires after the 100 ms hold, then every 450 ms while held: ~0.1 s, ~0.55 s, ~1.0 s
    assert taps.count("hotbar_next") in (2, 3) and set(taps) == {"hotbar_next"}
    taps, _, t = feed(mapper, flat(), 0.3, t)
    taps, _, t = feed(mapper, flat(hand__pinch_little=0.05), 0.3, t)
    assert taps == ["hotbar_prev"]


def test_index_pinch_still_right_click_and_not_a_tap(mapper):
    taps, cmds, _ = feed(mapper, flat(hand__pinch=0.1), 0.5)
    assert cmds.right_button and taps == []


def test_finger_pinch_ignored_in_fist_and_without_hand(mapper):
    taps, cmds, t = feed(mapper, flat(hand__curl_mean=0.8, hand__pinch_middle=0.05), 1.0)
    assert taps == []
    taps, _, _ = feed(mapper, flat(hand__present=0.0, hand__pinch_middle=0.05), 1.0, t)
    assert taps == []


def test_brief_touch_is_debounced(mapper):
    taps, _, t = feed(mapper, flat(hand__pinch_middle=0.05), 2 * DT)   # ~66 ms < 100 ms hold
    taps2, _, _ = feed(mapper, flat(), 0.3, t)
    assert taps == [] and taps2 == []


def test_no_taps_while_paused(mapper):
    mapper.set_paused(True)
    taps, cmds, _ = feed(mapper, flat(hand__pinch_middle=0.05), 1.0)
    assert taps == [] and cmds.paused


def test_tap_binding_validation():
    with pytest.raises(ValueError):
        Profile.from_dict({"bindings": [{"action": "inventory", "source": "gesture.pinch_middle",
                                         "repeat_ms": -1}]})
    with pytest.raises(ValueError):
        Profile.from_dict({"bindings": [{"action": "inventory", "source": "gesture.pinch_thumb"}]})
    p = Profile.from_dict({"gestures": {"finger_pinch": {"on": 0.2, "off": 0.35}}})
    assert p.gestures["finger_pinch"]["on"] == 0.2


# ------------------------------------------------------------------ outputs
def test_leonardo_sends_mouse_wheel_for_hotbar():
    from test_outputs import FakeClock, FakeSerial, C
    ser = FakeSerial()
    out = SerialLeonardoOutput(serial_obj=ser, clock=FakeClock())
    out.apply(C(taps=["hotbar_next", "hotbar_prev", "inventory"]))
    assert ser.lines() == ["W -1", "W 1", "K+e", "K-e"]


# ------------------------------------------------------------------ HUD
@pytest.mark.parametrize("detail", [False, True])
def test_hud_simple_and_detailed_views(detail):
    pytest.importorskip("cv2")
    from mcbio.hud import HudState, draw_hud
    from mcbio.types import Commands
    frame = np.zeros((720, 1280, 3), np.uint8)
    cmds = Commands(keys={"forward"}, left_button=True, taps=["inventory"])
    st = HudState(status="LIVE", fps=30, detail=detail, recent_taps=["hotbar_next"])
    out = draw_hud(frame, Features(), cmds, None, flat(), st)
    assert out.shape == frame.shape and out.any()
