"""Tests for mcbio.gestures (synthetic signals only)."""
import random

import pytest

from mcbio.gestures import CurlStateMachine, HoldTimer, Hysteresis

FPS = 30.0


def _transitions(states):
    return sum(1 for a, b in zip(states, states[1:]) if a != b)


def test_hysteresis_no_chatter_on_noisy_signal():
    rng = random.Random(0)
    det = Hysteresis(on=0.5, off=0.3)
    # Signal hovers right at the ON threshold with +-0.08 noise: a single threshold
    # would flip dozens of times; the hysteresis band must give one clean switch.
    out = [det.update(0.5 + rng.uniform(-0.08, 0.08), i / FPS) for i in range(300)]
    assert _transitions([False] + out) == 1 and out[-1] is True
    out = [det.update(0.4 + rng.uniform(-0.08, 0.08), (300 + i) / FPS) for i in range(300)]
    assert all(out)                         # stays ON inside the band
    out = [det.update(0.1 + rng.uniform(-0.08, 0.08), (600 + i) / FPS) for i in range(30)]
    assert _transitions(out) == 0 and out[-1] is False


def test_hysteresis_less_than_op():
    det = Hysteresis(on=0.3, off=0.45, op="<")
    assert det.update(0.5, 0.0) is False
    assert det.update(0.2, 0.1) is True
    assert det.update(0.4, 0.2) is True     # inside band
    assert det.update(0.5, 0.3) is False


def test_hysteresis_min_hold_debounces_spikes():
    det = Hysteresis(on=0.5, off=0.3, min_hold_ms=100)
    t = 0.0
    # 2-frame spike (66 ms) must be ignored.
    for v in (0.0, 0.9, 0.9, 0.0, 0.0):
        assert det.update(v, t) is False
        t += 1 / FPS
    # Sustained input turns on only after >= 100 ms.
    start = t
    while not det.update(0.9, t):
        t += 1 / FPS
    assert t - start >= 0.1 - 1e-9


def test_hysteresis_min_release_bridges_dropout():
    det = Hysteresis(on=0.5, off=0.3, min_release_ms=100)
    assert det.update(1.0, 0.0)
    assert det.update(0.0, 0.03) and det.update(0.0, 0.06)   # brief dropout bridged
    assert det.update(1.0, 0.09)
    assert det.update(0.0, 0.12) and not det.update(0.0, 0.25)


def test_hysteresis_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        Hysteresis(on=0.3, off=0.5, op=">")
    with pytest.raises(ValueError):
        Hysteresis(on=0.5, off=0.3, op="<")
    with pytest.raises(ValueError):
        Hysteresis(on=0.5, off=0.3, op="=")


def _curl_trace(seed=1):
    """open 0.5 s -> close 0.25 s -> hold 0.6 s -> open 0.25 s -> rest 0.5 s."""
    rng = random.Random(seed)
    segs = [("open", 0.5, 0.05, 0.05), ("close", 0.25, 0.05, 0.85),
            ("hold", 0.6, 0.85, 0.85), ("release", 0.25, 0.85, 0.05),
            ("rest", 0.5, 0.05, 0.05)]
    t, out = 0.0, []
    for name, dur, a, b in segs:
        n = int(round(dur * FPS))
        for i in range(n):
            out.append((name, t, a + (b - a) * (i + 1) / n + rng.gauss(0, 0.02)))
            t += 1 / FPS
    return out


def test_curl_state_machine_labels_open_close_hold_open():
    sm = CurlStateMachine()
    rows = [(seg, sm.update(theta, t), sm.is_closed) for seg, t, theta in _curl_trace()]
    by_seg = {}
    for seg, state, closed in rows:
        by_seg.setdefault(seg, []).append((state, closed))
    assert {s for s, _ in by_seg["open"]} == {"Rest"}
    assert "Flex" in {s for s, _ in by_seg["close"]}
    hold_states = [s for s, _ in by_seg["hold"]]
    assert hold_states[len(hold_states) // 3:].count("Hold") == len(hold_states[len(hold_states) // 3:])
    assert "Extend" in {s for s, _ in by_seg["release"]}
    assert by_seg["rest"][-1][0] == "Rest"
    # Fist latch: closed for the whole hold, open in rest, exactly one close/open cycle.
    assert all(c for _, c in by_seg["hold"])
    assert not any(c for _, c in by_seg["open"] + by_seg["rest"])
    closed = [c for _, _, c in rows]
    assert _transitions(closed) == 2


def test_curl_reset_and_duplicate_timestamps():
    sm = CurlStateMachine()
    sm.update(0.9, 0.0)
    assert sm.update(0.1, 0.0) == sm.state      # same t: ignored, no division by zero
    sm.reset()
    assert sm.state == "Rest" and not sm.is_closed


def test_hold_timer_fires_once_per_hold():
    ht = HoldTimer(1.0)
    fired = [ht.update(True, i / FPS) for i in range(90)]      # 3 s held
    assert sum(fired) == 1
    assert fired.index(True) == 30                              # at t = 1.0 s
    assert ht.update(False, 3.0) is False and ht.progress == 0.0
    fired = [ht.update(True, 3.1 + i / FPS) for i in range(35)]
    assert sum(fired) == 1                                      # re-armed after release


def test_hold_timer_short_holds_never_fire():
    ht = HoldTimer(0.6)
    t = 0.0
    for _ in range(10):                  # ten 0.3 s blinks
        for _ in range(9):
            assert not ht.update(True, t)
            t += 1 / FPS
        assert not ht.update(False, t)
        t += 1 / FPS
