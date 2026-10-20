"""Tests for mcbio.mapping using hand-built flat feature dicts."""
from pathlib import Path

import pytest

from mcbio.mapping import Mapper, Profile

PROFILES = Path(__file__).resolve().parent.parent / "profiles"
FPS = 30.0
DT = 1 / FPS


def neutral(face=True, hand=True, offhand=False) -> dict[str, float]:
    """A flat dict as features.flatten() would produce for a relaxed, neutral user."""
    d = {
        "face.present": float(face), "face.yaw": 0.0, "face.pitch": 0.0, "face.roll": 0.0,
        "face.nose_dx": 0.0, "face.nose_dy": 0.0, "face.scale_ratio": 0.0,
        "face.blend.jawOpen": 0.0, "face.blend.browInnerUp": 0.0,
        "face.blend.eyeBlinkLeft": 0.0, "face.blend.eyeBlinkRight": 0.0,
        "face.blend.mouthSmileLeft": 0.0, "face.blend.mouthSmileRight": 0.0,
        "face.blend.mouthPucker": 0.0, "face.blend.cheekPuff": 0.0,
        "hand.present": float(hand), "hand.palm_dx": 0.0, "hand.palm_dy": 0.0,
        "hand.curl_mean": 0.1, "hand.pinch": 1.0,
        "offhand.present": float(offhand), "offhand.curl_mean": 0.1 if offhand else 0.0,
        "offhand.pinch": 1.0 if offhand else 0.0,
        "hands.count": float(hand) + float(offhand),
    }
    for f in ("thumb", "index", "middle", "ring", "little"):
        d[f"hand.curl_{f}"] = 0.1
    return d


@pytest.fixture
def mapper():
    return Mapper(Profile.load(PROFILES / "face_hand.yaml"))


def run(mapper, flat, frames=5, t0=0.0):
    """Feed the same dict for several frames; return the last Commands and end time."""
    cmds = None
    for i in range(frames):
        cmds = mapper.update(flat, t0 + i * DT)
    return cmds, t0 + frames * DT


def test_neutral_input_gives_no_commands(mapper):
    cmds, _ = run(mapper, neutral(), frames=30)
    assert cmds.keys == set()
    assert cmds.mouse_dx == 0 and cmds.mouse_dy == 0
    assert not cmds.left_button and not cmds.right_button and not cmds.paused


def test_palm_up_is_forward(mapper):
    cmds, _ = run(mapper, {**neutral(), "hand.palm_dy": 0.5})
    assert cmds.keys == {"forward"}


def test_palm_up_right_is_forward_and_right(mapper):
    cmds, _ = run(mapper, {**neutral(), "hand.palm_dy": 0.5, "hand.palm_dx": 0.5})
    assert cmds.keys == {"forward", "right"}


def test_palm_hysteresis_keeps_key_inside_band(mapper):
    _, t = run(mapper, {**neutral(), "hand.palm_dy": 0.5})
    cmds, t = run(mapper, {**neutral(), "hand.palm_dy": 0.3}, t0=t)   # between off and on
    assert "forward" in cmds.keys
    cmds, _ = run(mapper, {**neutral(), "hand.palm_dy": 0.2}, t0=t)
    assert "forward" not in cmds.keys


def test_opposing_keys_never_both_held():
    prof = Profile.from_dict({"bindings": [
        {"action": "forward", "source": "face.scale_ratio", "op": ">", "on": 0.1, "off": 0.05},
        {"action": "back", "source": "face.pitch", "op": ">", "on": 5, "off": 3},
        {"action": "left", "source": "face.roll", "op": "<", "on": -5, "off": -3},
        {"action": "right", "source": "face.yaw", "op": ">", "on": 5, "off": 3},
    ]})
    m = Mapper(prof)
    # forward pushed far past its threshold, back only just -> forward wins.
    cmds, t = run(m, {**neutral(), "face.scale_ratio": 0.5, "face.pitch": 6.0,
                      "face.roll": -20.0, "face.yaw": 6.0})
    assert cmds.keys == {"forward", "left"}
    cmds, _ = run(m, {**neutral(), "face.scale_ratio": 0.11, "face.pitch": 30.0,
                      "face.roll": -6.0, "face.yaw": 40.0}, t0=t)
    assert cmds.keys == {"back", "right"}


def test_jaw_open_is_jump(mapper):
    cmds, _ = run(mapper, {**neutral(), "face.blend.jawOpen": 0.6})
    assert "jump" in cmds.keys


def test_fist_holds_left_button_and_pinch_right(mapper):
    t = 0.0
    for i in range(10):                      # close the hand over ~0.33 s
        cmds = mapper.update({**neutral(), "hand.curl_mean": 0.1 + 0.08 * (i + 1)}, t)
        t += DT
    cmds, t = run(mapper, {**neutral(), "hand.curl_mean": 0.9}, frames=10, t0=t)
    assert cmds.left_button and not cmds.right_button   # fist is NOT a pinch
    cmds, t = run(mapper, neutral(), frames=15, t0=t)
    assert not cmds.left_button
    cmds, _ = run(mapper, {**neutral(), "hand.pinch": 0.1}, frames=6, t0=t)  # > 80 ms
    assert cmds.right_button


def test_head_yaw_beyond_deadzone_moves_mouse_proportional_to_dt(mapper):
    flat = {**neutral(), "face.yaw": 16.0, "face.pitch": 0.0}
    mapper.update(flat, 0.0)
    a = mapper.update(flat, 0.010).mouse_dx
    b = mapper.update(flat, 0.030).mouse_dx      # dt twice as long
    assert a > 0 and b == pytest.approx(2 * a)
    left = mapper.update({**flat, "face.yaw": -16.0}, 0.040).mouse_dx
    assert left == pytest.approx(-a)
    inside = mapper.update({**flat, "face.yaw": 5.0}, 0.050).mouse_dx
    assert inside == 0


def test_head_pitch_up_moves_mouse_up(mapper):
    mapper.update({**neutral(), "face.pitch": 20.0}, 0.0)
    assert mapper.update({**neutral(), "face.pitch": 20.0}, DT).mouse_dy < 0


def test_look_speed_is_capped_per_frame(mapper):
    mapper.update({**neutral(), "face.yaw": 90.0}, 0.0)
    cap = mapper.profile.look.max_px_per_frame
    assert mapper.update({**neutral(), "face.yaw": 90.0}, 0.1).mouse_dx == cap


def test_face_absent_means_no_look_and_no_jump(mapper):
    flat = {**neutral(face=False), "face.yaw": 30.0, "face.blend.jawOpen": 0.9}
    cmds, _ = run(mapper, flat)
    assert cmds.mouse_dx == 0 and cmds.mouse_dy == 0 and "jump" not in cmds.keys


def test_hand_absent_releases_hand_actions(mapper):
    _, t = run(mapper, {**neutral(), "hand.palm_dy": 0.6})
    # Hand lost: flatten() reports zeros and present=0 -> everything hand-driven released.
    cmds, _ = run(mapper, {**neutral(hand=False), "hand.pinch": 0.0}, frames=1, t0=t)
    assert cmds.keys == set() and not cmds.right_button


def test_pause_gesture_toggles_and_paused_is_idle(mapper):
    both_open = {**neutral(offhand=True), "hand.curl_mean": 0.05, "offhand.curl_mean": 0.05,
                 "hand.palm_dy": 0.6, "face.yaw": 30.0}
    cmds, t = run(mapper, both_open, frames=int(1.2 * FPS))
    assert cmds.paused
    assert cmds.keys == set() and cmds.mouse_dx == 0 and not cmds.left_button
    # Still holding: must NOT un-pause again (fires once per hold).
    cmds, t = run(mapper, both_open, frames=int(1.5 * FPS), t0=t)
    assert cmds.paused
    # While paused, normal input is ignored.
    cmds, t = run(mapper, {**neutral(), "hand.palm_dy": 0.6}, frames=10, t0=t)
    assert cmds.paused and cmds.keys == set()
    # Show both palms again for 1 s -> resume.
    cmds, t = run(mapper, both_open, frames=int(1.2 * FPS), t0=t)
    assert not cmds.paused
    cmds, _ = run(mapper, {**neutral(), "hand.palm_dy": 0.6}, t0=t)
    assert cmds.keys == {"forward"}


def test_short_both_open_does_not_pause(mapper):
    both_open = {**neutral(offhand=True), "hand.curl_mean": 0.05, "offhand.curl_mean": 0.05}
    cmds, _ = run(mapper, both_open, frames=int(0.5 * FPS))
    assert not cmds.paused


def test_external_pause_toggle(mapper):
    mapper.toggle_pause()
    cmds, _ = run(mapper, {**neutral(), "hand.palm_dy": 0.6})
    assert cmds.paused and cmds.keys == set()


def test_debug_contains_binding_values(mapper):
    cmds, _ = run(mapper, {**neutral(), "hand.palm_dy": 0.5})
    dbg = cmds.debug
    assert {"bindings", "look", "curl_state", "mode", "paused", "gestures"} <= set(dbg)
    fwd = [b for b in dbg["bindings"] if b["action"] == "forward"][0]
    assert fwd["value"] == 0.5 and fwd["on"] is True


@pytest.mark.parametrize("name", ["face_hand", "face_only", "hand_only"])
def test_all_profiles_load_and_idle_on_neutral(name):
    m = Mapper(Profile.load(PROFILES / f"{name}.yaml"))
    for kw in ({}, {"face": False}, {"hand": False}, {"offhand": True}):
        m = Mapper(m.profile)
        cmds, _ = run(m, neutral(**kw), frames=30)
        assert cmds.keys == set() and cmds.mouse_dx == 0 and cmds.mouse_dy == 0
        assert not cmds.left_button and not cmds.right_button and not cmds.paused


def test_face_only_mode_toggle_switches_walk_and_look():
    m = Mapper(Profile.load(PROFILES / "face_only.yaml"))
    lean_and_turn = {**neutral(hand=False), "face.scale_ratio": 0.15, "face.yaw": 20.0}
    cmds, t = run(m, lean_and_turn)
    assert m.mode == "walk" and cmds.keys == {"forward"} and cmds.mouse_dx == 0
    blink = {**neutral(hand=False), "face.blend.eyeBlinkLeft": 0.9, "face.blend.eyeBlinkRight": 0.9}
    _, t = run(m, blink, frames=int(0.8 * FPS), t0=t)
    assert m.mode == "look"
    cmds, _ = run(m, lean_and_turn, t0=t)
    assert cmds.keys == set() and cmds.mouse_dx > 0


def test_face_only_brow_and_pucker_click():
    m = Mapper(Profile.load(PROFILES / "face_only.yaml"))
    cmds, _ = run(m, {**neutral(hand=False), "face.blend.browInnerUp": 0.7,
                      "face.blend.cheekPuff": 0.7}, frames=8)
    assert cmds.left_button and cmds.right_button


def test_invalid_profiles_are_rejected():
    with pytest.raises(ValueError):
        Profile.from_dict({"bindings": [{"action": "fly", "source": "face.yaw", "on": 1, "off": 0}]})
    with pytest.raises(ValueError):
        Profile.from_dict({"bindings": [{"action": "jump", "source": "face.yaw"}]})
    with pytest.raises(ValueError):
        Profile.from_dict({"bindings": [{"action": "jump", "source": "gesture.wave"}]})
    with pytest.raises(ValueError):   # mode used without mode_toggle
        Profile.from_dict({"bindings": [{"action": "jump", "source": "gesture.fist", "mode": "look"}]})
