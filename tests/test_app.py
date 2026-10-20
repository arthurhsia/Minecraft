"""End-to-end tests of the app pipeline in synthetic + dry-run mode (no camera/mediapipe).

Skipped automatically if the other agents' modules (features / mapping / outputs /
calibration) are not importable yet.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcbio.features")
pytest.importorskip("mcbio.mapping")
pytest.importorskip("mcbio.outputs")
pytest.importorskip("mcbio.calibration")

from mcbio.app import ARMING, LIVE, PAUSED, App, AppConfig, CalibrationFlow, config_from_args  # noqa: E402
from mcbio.synthetic import SyntheticFeatures  # noqa: E402
from mcbio.types import Commands  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class RecordingBackend:
    """Wraps DryRunOutput and records every apply / release_all / close call."""

    def __init__(self) -> None:
        from mcbio.outputs import DryRunOutput
        self.inner = DryRunOutput()
        self.applied: list[Commands] = []
        self.release_calls = 0
        self.closed = False

    def apply(self, cmds: Commands) -> None:
        self.applied.append(cmds)
        self.inner.apply(cmds)

    def release_all(self) -> None:
        self.release_calls += 1
        self.inner.release_all()

    def close(self) -> None:
        self.closed = True
        self.inner.close()


def _cfg(**kw) -> AppConfig:
    base = dict(backend="dry", synthetic=True, hud=False, hotkey=False, quiet=True, frames=200)
    base.update(kw)
    return AppConfig(**base)


def test_pipeline_synthetic_dry_200_frames() -> None:
    be = RecordingBackend()
    app = App(_cfg(), backend=be)
    summary = app.run()

    assert summary["frames"] == 200 and len(be.applied) == 200
    # countdown: nothing sent during the first 3 s (90 frames at 30 fps)
    assert all(c.paused and not c.keys and not c.left_button for c in be.applied[:89])
    live = [c for c in be.applied if not c.paused]
    assert live, "app never went LIVE after the countdown"
    # the scripted user moved forward and looked around within ~6.6 s
    assert any("forward" in c.keys for c in live)
    assert any(abs(c.mouse_dx) > 0 for c in live)
    assert "forward" in summary["actions"]
    # safety: everything released at the end
    assert be.release_calls >= 1 and be.closed
    assert be.inner.held_keys == set() and be.inner.held_buttons == set()
    assert summary["released_cleanly"]
    assert summary["fps"] == pytest.approx(30.0, abs=0.5)
    assert summary["latency_ms_p95"] >= 0.0


def test_pause_gesture_and_hotkey_toggle() -> None:
    be = RecordingBackend()
    app = App(_cfg(frames=700), backend=be)
    states: list[str] = []
    orig = app._after_frame

    def spy(*a, **k):
        states.append(app.state)
        return orig(*a, **k)

    app._after_frame = spy  # type: ignore[method-assign]
    summary = app.run()
    # scripted both-palms gesture at ~12-14 s pauses, again at ~16-18 s resumes (via ARMING)
    assert PAUSED in states and states[-1] in (LIVE, ARMING)
    i = states.index(PAUSED)
    assert ARMING in states[i:]
    assert summary["pause_toggles"] >= 2
    # nothing pressed while paused
    for st, c in zip(states, be.applied):
        if st == PAUSED:
            assert not c.keys and not c.left_button and not c.right_button and c.mouse_dx == 0

    # F8 hotkey path: toggle_pause from LIVE -> PAUSED -> ARMING
    be2 = RecordingBackend()
    app2 = App(_cfg(frames=1), backend=be2)
    app2.run()
    app2.state = LIVE
    app2.toggle_pause(10.0)
    assert app2.state == PAUSED and be2.release_calls >= 2
    app2.toggle_pause(11.0)
    assert app2.state == ARMING and app2.arm_until == pytest.approx(14.0)


def test_release_all_on_exception() -> None:
    be = RecordingBackend()

    class Boom:
        mirrored = True

        def __init__(self) -> None:
            self.s = SyntheticFeatures(0)
            self.n = 0

        def grab(self):
            self.n += 1
            if self.n > 150:
                raise KeyboardInterrupt      # simulates Ctrl-C mid-run
            f = self.s.next()
            return None, f.t, f

        def calibration(self):
            return self.s.calibration()

        def close(self) -> None:
            pass

    app = App(_cfg(frames=None), backend=be, source=Boom())
    summary = app.run()
    assert summary["frames"] == 150
    assert be.release_calls >= 1 and be.closed and be.inner.held_keys == set()


def test_calibration_flow_on_synthetic_features() -> None:
    synth = SyntheticFeatures(1, mode="responsive", false_rate_per_s=0.0)
    flow = CalibrationFlow(dominant_hand="Right", countdown_s=1.0, neutral_s=1.0, range_s=1.0)
    prompts = set()
    while not flow.done:
        f = synth.next()
        flow.update(f, f.t)
        prompts.add(flow.prompt[0])
        assert f.t < 10
    c = flow.result
    true = synth.calibration()
    assert c is not None
    assert c.neutral_yaw == pytest.approx(true.neutral_yaw, abs=1.0)
    assert c.neutral_palm[1] == pytest.approx(true.neutral_palm[1], abs=0.02)
    assert any("comfortable edge" in p for p in prompts)


def test_protocol_and_logging(tmp_path: Path) -> None:
    be = RecordingBackend()
    app = App(_cfg(frames=None, protocol=True, protocol_trials=6, log_dir=str(tmp_path),
                   subject="T1", seed=2), backend=be)
    summary = app.run()
    assert summary["cues_done"] == 6
    (d,) = list(tmp_path.iterdir())
    for name in ("frames.csv", "events.csv", "cues.csv", "session.json"):
        assert (d / name).exists(), name
    meta = json.loads((d / "session.json").read_text())
    assert meta["subject"] == "T1" and meta["backend"] == "dry" and meta["protocol"] is True
    assert meta["summary"]["cues_done"] == 6

    # analysis script runs on it
    out = tmp_path / "analysis"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_session.py"), str(tmp_path),
                        "--out", str(out), "--no-plots"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert (out / "summary_subjects.csv").exists() and "T1" in r.stdout


def test_cli_parsing_and_headless_run() -> None:
    cfg = config_from_args(["--synthetic", "--dry-run", "--no-hud", "--frames", "5", "--size", "640x480"])
    assert cfg.backend == "dry" and not cfg.hud and cfg.frames == 5 and (cfg.width, cfg.height) == (640, 480)
    r = subprocess.run([sys.executable, "-m", "mcbio", "--synthetic", "--dry-run", "--no-hud",
                        "--frames", "120"], capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "mcbio summary" in r.stdout and "released: yes" in r.stdout
