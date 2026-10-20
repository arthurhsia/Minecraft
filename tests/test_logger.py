"""Tests for mcbio.logger (CSV/JSON outputs, protocol reaction-time logic) and HUD drawing."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mcbio.logger import CUE_FIELDS, Protocol, SessionLogger, active_actions
from mcbio.types import Commands, FaceFeatures, Features, HandFeatures, HeadPose


def _rows(p: Path) -> list[dict[str, str]]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- actions
def test_active_actions_mapping() -> None:
    c = Commands(keys={"forward", "jump"}, mouse_dx=-3.0, mouse_dy=2.0,
                 left_button=True, right_button=False)
    assert active_actions(c) == {"forward", "jump", "attack", "look_left", "look_down"}
    c2 = Commands(right_button=True, mouse_dx=0.2)       # below look threshold
    assert active_actions(c2) == {"interact"}
    assert active_actions(Commands(keys={"forward"}, paused=True)) == set()
    assert active_actions(None) == set()


# ----------------------------------------------------------------------------- logger
def test_session_logger_writes_csv_and_json(tmp_path: Path) -> None:
    log = SessionLogger(tmp_path, "S01", meta={"profile": "face_hand", "backend": "dry"})
    flat = {"face.yaw": 1.5, "hand.palm_dx": 0.2, "face.blend.jawOpen": 0.1}
    seq = [
        (0.00, Commands.idle(paused=True), True),
        (0.03, Commands(keys={"forward"}), False),
        (0.07, Commands(keys={"forward"}, left_button=True, taps=["inventory"]), False),
        (0.10, Commands(left_button=True), False),
        (0.13, Commands(), False),
    ]
    for t, c, paused in seq:
        log.log_frame(t, latency_ms=12.5, fps=30.0, paused=paused, cmds=c, flat=flat)
    log.log_cue({"trial": 1, "cue": "JUMP", "expected": "jump", "cue_t": 0.0, "window_s": 3.0,
                 "first_action": "jump", "first_action_t": 0.5, "rt_ms": 500.0, "correct": True,
                 "match_rt_ms": 500.0, "false_activations": 0, "wrong_actions": "", "outcome": "hit"})
    log.close({"frames": 5}, t=0.2)
    d = log.path
    assert d.parent == tmp_path and d.name.startswith("S01_")

    frames = _rows(d / "frames.csv")
    assert len(frames) == 5
    assert frames[0]["paused"] == "1" and frames[0]["actions"] == ""
    assert frames[2]["actions"] == "attack forward"
    assert frames[2]["taps"] == "inventory"
    assert float(frames[1]["face.yaw"]) == pytest.approx(1.5)
    assert frames[1]["hand.curl_mean"] == ""                    # missing key -> blank
    assert float(frames[1]["latency_ms"]) == pytest.approx(12.5)

    ev = [(r["action"], r["state"], float(r["t"])) for r in _rows(d / "events.csv")]
    assert ("paused", "off", 0.03) in ev
    assert ("forward", "on", 0.03) in ev and ("forward", "off", 0.10) in ev
    assert ("attack", "on", 0.07) in ev and ("attack", "off", 0.13) in ev
    assert ("inventory", "tap", 0.07) in ev

    cues = _rows(d / "cues.csv")
    assert list(cues[0]) == CUE_FIELDS and cues[0]["correct"] == "1"

    meta = json.loads((d / "session.json").read_text())
    assert meta["subject"] == "S01" and meta["profile"] == "face_hand" and meta["backend"] == "dry"
    assert meta["frames"] == 5 and meta["summary"] == {"frames": 5}
    assert "start_time" in meta and "end_time" in meta


def test_logger_never_overwrites(tmp_path: Path) -> None:
    a = SessionLogger(tmp_path, "X", session_name=None)
    b = SessionLogger(tmp_path, "X", session_name=None)
    assert a.path != b.path
    a.close()
    b.close()


# ----------------------------------------------------------------------------- protocol
def _drive(proto: Protocol, script: dict[float, set[str]], t_end: float, dt: float = 1 / 30):
    """Run the protocol with actions given as {start_time: actions} step changes."""
    rows = []
    acts: set[str] = set()
    t = 0.0
    keys = sorted(script)
    while t < t_end:
        for k in keys:
            if abs(t - k) < dt / 2:
                acts = script[k]
        rows += proto.update(round(t, 6), acts)
        t += dt
    return rows


def test_protocol_reaction_time_correct_and_wrong() -> None:
    cues = {"JUMP": "jump", "REST": None}
    seen: list[tuple[str, float]] = []
    p = Protocol(seed=1, cues=cues, n_trials=50, interval=(2.0, 2.0), rest_interval=(3.0, 3.0),
                 gap=0.0, lead_in=0.0, rest_per_block=0, on_cue=lambda c, t: seen.append((c, t)))
    # rest_per_block=0 -> only JUMP cues; first cue appears at t=0, window 2 s
    rows = _drive(p, {0.5: {"jump"}, 1.0: set(), 2.6: {"forward"}, 3.0: {"forward", "jump"},
                      3.5: set()}, t_end=6.1)
    assert seen[0] == ("JUMP", 0.0)
    first, second = rows[0], rows[1]
    assert first["correct"] and first["outcome"] == "hit"
    assert first["rt_ms"] == pytest.approx(500.0, abs=1.0)
    # second cue starts at ~2.0: the first onset is "forward" (wrong), jump comes later
    assert not second["correct"] and second["outcome"] == "wrong"
    assert second["first_action"] == "forward"
    assert second["rt_ms"] == pytest.approx(600.0, abs=40.0)
    assert second["match_rt_ms"] == pytest.approx(1000.0, abs=40.0)
    assert "forward" in second["wrong_actions"]
    # third cue: nothing -> miss
    assert rows[2]["outcome"] == "miss" and rows[2]["rt_ms"] is None


def test_protocol_rest_false_activations_and_held_actions() -> None:
    p = Protocol(seed=0, cues={"REST": None}, n_trials=2, rest_interval=(3.0, 3.0),
                 gap=0.0, lead_in=0.0, rest_per_block=1)
    # "forward" already held when REST starts is NOT counted (no onset); two later onsets are
    rows = _drive(p, {0.0: {"forward"}, 1.0: set(), 1.5: {"jump"}, 1.7: set(), 2.0: {"attack"},
                      2.2: set()}, t_end=7.0)
    assert len(rows) == 2 and p.finished
    assert rows[0]["false_activations"] == 2 and rows[0]["outcome"] == "false" and not rows[0]["correct"]
    assert rows[1]["false_activations"] == 0 and rows[1]["outcome"] == "clean" and rows[1]["correct"]


def test_protocol_pause_discards_cue_and_blocks_are_balanced() -> None:
    p = Protocol(seed=3, n_trials=100, lead_in=0.0, gap=0.0)
    p.update(0.0, set())
    assert p.current is not None
    p.update(0.1, set(), live=False)            # pause -> discarded
    assert p.current is None and p.completed == 0
    # one full block contains each action cue once and REST twice
    labels: list[str] = []
    p2 = Protocol(seed=5, n_trials=11, lead_in=0.0, gap=0.0, on_cue=lambda c, t: labels.append(c))
    t = 0.0
    while not p2.finished:
        p2.update(t, set())
        t += 0.1
    assert len(labels) == 11
    assert labels.count("REST") == 2 and len(set(labels)) == 10


# ----------------------------------------------------------------------------- HUD
def _features() -> Features:
    from mcbio.synthetic import hand_landmarks
    hand = HandFeatures(present=True, palm=(0.6, 0.6), curl=(0.2, 0.9, 0.9, 0.9, 0.9),
                        pinch=0.5, scale=0.1, landmarks=hand_landmarks((0.6, 0.6), 0.1, (0.2, 0.9, 0.9, 0.9, 0.9)))
    face = FaceFeatures(present=True, head=HeadPose(10, -3, 0), blend={"jawOpen": 0.6},
                        nose=(0.5, 0.4), scale=0.08, landmarks=[(0.45, 0.35, 0), (0.55, 0.35, 0)])
    return Features(t=1.0, face=face, hands=[hand])


@pytest.mark.parametrize("status", ["LIVE", "PAUSED", "ARMING", "CALIBRATING"])
@pytest.mark.parametrize("shape", [(720, 1280, 3), (480, 640, 3), (240, 320, 3)])
def test_hud_draw_returns_same_shape(status: str, shape: tuple[int, int, int]) -> None:
    from mcbio.hud import HudState, draw_hud
    frame = np.full(shape, 40, np.uint8)
    cmds = Commands(keys={"forward", "jump"}, mouse_dx=5, left_button=True, taps=["inventory"],
                    debug={"curl_state": "Hold", "move_deadzone": 0.35, "look_deadzone_deg": 6.0})
    flat = {"hand.present": 1.0, "hand.palm_dx": 0.4, "hand.palm_dy": 0.8, "face.present": 1.0,
            "face.yaw": 10.0, "face.pitch": -3.0, "hand.curl_mean": 0.9, "hand.pinch": 0.5,
            "face.blend.jawOpen": 0.6}
    st = HudState(status=status, fps=29.7, latency_ms=23.0, countdown=2.4, cue="JUMP",
                  cue_progress=0.3, prompt=("Sit relaxed and look at the screen", "rest your hand"),
                  prompt_progress=0.5, prompt_count=3, profile="face_hand", backend="dry",
                  neutral_palm=(0.55, 0.6), palm_range=0.12, message="hello")
    before = frame.copy()
    out = draw_hud(frame, _features(), cmds, cmds.debug, flat, st, mirror=True)
    assert out.shape == shape and out.dtype == np.uint8
    assert np.array_equal(frame, before), "input frame must not be modified"
    assert not np.array_equal(out, before), "something must be drawn"


def test_hud_handles_missing_everything() -> None:
    from mcbio.hud import draw_hud
    frame = np.zeros((360, 640, 3), np.uint8)
    out = draw_hud(frame, None, None)
    assert out.shape == frame.shape
    out2 = draw_hud(frame, Features(), Commands.idle(paused=True))
    assert out2.shape == frame.shape


def test_hud_individual_widgets() -> None:
    from mcbio import hud
    f = np.zeros((720, 1280, 3), np.uint8)
    for fn in (lambda: hud.draw_joystick(f, (10, 400), 200, 2.0, -2.0, True, 0.35, 0.25, active={"forward"}),
               lambda: hud.draw_head_gauge(f, (230, 400), 200, -40, 40, True),
               lambda: hud.draw_gesture_panel(f, (800, 400), 400, "Flex", 0.5, 0.2, 0.7, 0.1),
               lambda: hud.draw_banner(f, "PAUSED", "sub"),
               lambda: hud.draw_prompt(f, "Title " * 20, "subtitle " * 20, 0.5, 2),
               lambda: hud.draw_cue(f, "LOOK LEFT", 0.5, "trial 1/30"),
               lambda: hud.draw_action_chips(f, {"forward", "attack"}, ["hotbar_next"])):
        assert fn().shape == (720, 1280, 3)
    w = hud.HudWindow("test")     # headless OpenCV here: must degrade, not crash
    assert w.show(f) in (-1, 27) or w.available
    w.close()
