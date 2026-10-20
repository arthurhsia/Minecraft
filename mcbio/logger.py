"""Session logging and the cue/reaction-time experiment protocol.

Files written per session directory (``<log_dir>/<subject>_<YYYYmmdd-HHMMSS>/``):

``session.json``  metadata: subject, profile, backend, start/end time, frame count, summary.
``frames.csv``    one row per frame: t, latency_ms, fps, paused, active actions, mouse, key features.
``events.csv``    action on/off edges (and taps, pause on/off): t, action, state.
``cues.csv``      (``--protocol`` only) one row per cue: reaction time, correctness, false activations.

The protocol (:class:`Protocol`) shows random cues (FORWARD, BACK, LEFT, RIGHT, JUMP,
ATTACK, INTERACT, LOOK LEFT, LOOK RIGHT and REST) for random 2-4 s windows, in shuffled
balanced blocks. For each cue it records the *first new action onset* (rising edge)
after the cue appeared: reaction time = onset t - cue t; correct = that first onset is
the expected action. During REST every onset is a false activation.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from mcbio.types import Commands

#: Logical action names used in logs, chips and the protocol.
ACTIONS: tuple[str, ...] = ("forward", "back", "left", "right", "jump", "sneak", "sprint",
                            "attack", "interact", "look_left", "look_right", "look_up", "look_down")

#: Cue label -> expected action (None = REST: no action expected).
CUES: dict[str, str | None] = {
    "FORWARD": "forward", "BACK": "back", "LEFT": "left", "RIGHT": "right",
    "JUMP": "jump", "ATTACK": "attack", "INTERACT": "interact",
    "LOOK LEFT": "look_left", "LOOK RIGHT": "look_right", "REST": None,
}

#: Flat feature keys copied into frames.csv (missing keys are written blank).
FRAME_FEATURES: tuple[str, ...] = (
    "face.present", "face.yaw", "face.pitch", "face.roll",
    "hand.present", "hand.palm_dx", "hand.palm_dy", "hand.curl_mean", "hand.pinch",
    "face.blend.jawOpen", "face.blend.browInnerUp", "hands.count",
)


def active_actions(cmds: Commands | None, look_threshold_px: float = 0.5) -> set[str]:
    """Logical actions active in ``cmds`` (empty while paused).

    Keys map to themselves; ``left_button`` -> ``attack``; ``right_button`` -> ``interact``;
    mouse motion beyond ``look_threshold_px`` per frame -> ``look_left/right/up/down``
    (screen convention: +dx = right, +dy = down).
    """
    if cmds is None or cmds.paused:
        return set()
    acts = {k for k in cmds.keys}
    if cmds.left_button:
        acts.add("attack")
    if cmds.right_button:
        acts.add("interact")
    if cmds.mouse_dx < -look_threshold_px:
        acts.add("look_left")
    elif cmds.mouse_dx > look_threshold_px:
        acts.add("look_right")
    if cmds.mouse_dy < -look_threshold_px:
        acts.add("look_up")
    elif cmds.mouse_dy > look_threshold_px:
        acts.add("look_down")
    return acts


def _fmt(v: Any) -> Any:
    """Compact CSV formatting for floats/bools."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return v


class SessionLogger:
    """Writes frames.csv / events.csv / cues.csv / session.json for one session.

    Args:
        root: parent directory (e.g. ``sessions/``); a per-session subdirectory is created.
        subject: participant ID (used in the directory name and metadata).
        meta: extra metadata for session.json (profile, backend, argv, ...).
        session_name: override the subdirectory name.
        flush_every: flush CSV files every N frames (and always on close).
    """

    def __init__(self, root: str | os.PathLike, subject: str = "anon",
                 meta: dict[str, Any] | None = None, session_name: str | None = None,
                 flush_every: int = 30) -> None:
        self.subject = subject or "anon"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = session_name or f"{_safe(self.subject)}_{stamp}"
        path = Path(root) / name
        n = 1
        while path.exists() and session_name is None:      # never overwrite a session
            n += 1
            path = Path(root) / f"{name}-{n}"
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.flush_every = max(1, flush_every)
        self.meta: dict[str, Any] = {
            "subject": self.subject,
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        self.meta.update(meta or {})
        self._frames_f = open(path / "frames.csv", "w", newline="")
        self._events_f = open(path / "events.csv", "w", newline="")
        self._cues_f = None
        self._cues_w: csv.DictWriter | None = None
        self._frames = csv.writer(self._frames_f)
        self._events = csv.writer(self._events_f)
        self._frames.writerow(["t", "latency_ms", "fps", "paused", "actions", "mouse_dx",
                               "mouse_dy", "taps", "cue", *FRAME_FEATURES])
        self._events.writerow(["t", "action", "state"])
        self._prev_actions: set[str] = set()
        self._prev_paused: bool | None = None
        self.n_frames = 0
        self.n_events = 0
        self.n_cues = 0
        self.closed = False
        self._write_meta()

    # ------------------------------------------------------------------ logging
    def log_frame(self, t: float, latency_ms: float, fps: float, paused: bool,
                  cmds: Commands | None, flat: dict[str, float] | None = None,
                  cue: str | None = None) -> set[str]:
        """Log one frame; emits on/off events for action edges. Returns active actions."""
        actions = set() if paused else active_actions(cmds)
        flat = flat or {}
        taps = list(cmds.taps) if (cmds is not None and not paused) else []
        self._frames.writerow([
            f"{t:.4f}", f"{latency_ms:.2f}", f"{fps:.1f}", int(bool(paused)),
            " ".join(sorted(actions)),
            _fmt(float(cmds.mouse_dx)) if cmds and not paused else "0",
            _fmt(float(cmds.mouse_dy)) if cmds and not paused else "0",
            " ".join(taps), cue or "",
            *[_fmt(float(flat[k])) if k in flat else "" for k in FRAME_FEATURES],
        ])
        # events: pause edges, action edges, taps
        if self._prev_paused is None or bool(paused) != self._prev_paused:
            if self._prev_paused is not None:
                self.log_event(t, "paused", "on" if paused else "off")
            self._prev_paused = bool(paused)
        for a in sorted(actions - self._prev_actions):
            self.log_event(t, a, "on")
        for a in sorted(self._prev_actions - actions):
            self.log_event(t, a, "off")
        for tap in taps:
            self.log_event(t, tap, "tap")
        self._prev_actions = actions
        self.n_frames += 1
        if self.n_frames % self.flush_every == 0:
            self.flush()
        return actions

    def log_event(self, t: float, action: str, state: str) -> None:
        """Append one row to events.csv."""
        self._events.writerow([f"{t:.4f}", action, state])
        self.n_events += 1

    def log_cue(self, row: dict[str, Any]) -> None:
        """Append one protocol result row (see :meth:`Protocol.update`) to cues.csv."""
        if self._cues_w is None:
            self._cues_f = open(self.path / "cues.csv", "w", newline="")
            self._cues_w = csv.DictWriter(self._cues_f, fieldnames=CUE_FIELDS, extrasaction="ignore")
            self._cues_w.writeheader()
        self._cues_w.writerow({k: _fmt(v) if v is not None else "" for k, v in row.items()})
        self.n_cues += 1

    def flush(self) -> None:
        for f in (self._frames_f, self._events_f, self._cues_f):
            if f is not None and not f.closed:
                f.flush()

    def close(self, summary: dict[str, Any] | None = None, t: float | None = None) -> None:
        """Close all files; turn off any still-active actions; finalize session.json."""
        if self.closed:
            return
        if t is not None:
            for a in sorted(self._prev_actions):
                self.log_event(t, a, "off")
            self._prev_actions = set()
        self.meta["end_time"] = datetime.now().isoformat(timespec="seconds")
        self.meta["frames"] = self.n_frames
        self.meta["events"] = self.n_events
        self.meta["cues"] = self.n_cues
        if summary is not None:
            self.meta["summary"] = summary
        for f in (self._frames_f, self._events_f, self._cues_f):
            if f is not None and not f.closed:
                f.close()
        self._write_meta()
        self.closed = True

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write_meta(self) -> None:
        tmp = self.path / "session.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2, default=str))
        tmp.replace(self.path / "session.json")


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


# ======================================================================================
# Protocol
# ======================================================================================
CUE_FIELDS: list[str] = ["trial", "cue", "expected", "cue_t", "window_s", "first_action",
                         "first_action_t", "rt_ms", "correct", "match_rt_ms",
                         "false_activations", "wrong_actions", "outcome"]


@dataclass
class _Trial:
    trial: int
    cue: str
    expected: str | None
    t_on: float
    window: float
    first_action: str | None = None
    first_action_t: float | None = None
    match_t: float | None = None
    false_activations: int = 0
    wrong: list[str] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        rt = None if self.first_action_t is None else (self.first_action_t - self.t_on) * 1000.0
        mrt = None if self.match_t is None else (self.match_t - self.t_on) * 1000.0
        if self.expected is None:
            correct = self.false_activations == 0
            outcome = "clean" if correct else "false"
        else:
            correct = self.first_action == self.expected
            outcome = "hit" if correct else ("wrong" if self.first_action else "miss")
        return {"trial": self.trial, "cue": self.cue, "expected": self.expected or "",
                "cue_t": self.t_on, "window_s": self.window, "first_action": self.first_action or "",
                "first_action_t": self.first_action_t, "rt_ms": rt, "correct": bool(correct),
                "match_rt_ms": mrt, "false_activations": self.false_activations,
                "wrong_actions": " ".join(self.wrong), "outcome": outcome}


class Protocol:
    """Random-cue reaction-time protocol.

    Call :meth:`update` every frame with the frame time and the set of active actions.
    Cues come in shuffled blocks containing every cue once (REST ``rest_per_block`` times),
    so each subject gets a balanced design. A cue is displayed for a random window
    (``interval`` s; REST uses ``rest_interval``), followed by a short blank ``gap``.

    Args:
        seed: RNG seed (use a different seed per subject, or the subject ID).
        cues: cue label -> expected action (None for REST).
        n_trials: total number of cues; :attr:`finished` becomes True afterwards.
        interval: (min, max) response window for action cues, seconds.
        rest_interval: (min, max) duration of REST periods, seconds.
        gap: blank time between cues, seconds.
        lead_in: wait after start/resume before the first cue, seconds.
        rest_per_block: number of REST cues per block.
        on_cue: optional callback ``(label, t)`` fired when a cue appears.
    """

    def __init__(self, seed: int | str = 0, cues: dict[str, str | None] | None = None,
                 n_trials: int = 30, interval: tuple[float, float] = (2.0, 4.0),
                 rest_interval: tuple[float, float] = (3.0, 6.0), gap: float = 0.5,
                 lead_in: float = 1.5, rest_per_block: int = 2,
                 on_cue: Callable[[str, float], None] | None = None) -> None:
        self.rng = random.Random(seed)
        self.cues = dict(cues or CUES)
        self.n_trials = n_trials
        self.interval = interval
        self.rest_interval = rest_interval
        self.gap = gap
        self.lead_in = lead_in
        self.rest_per_block = rest_per_block
        self.on_cue = on_cue
        self._queue: list[str] = []
        self._trial: _Trial | None = None
        self._next_start: float | None = None
        self._prev: set[str] = set()
        self._last_cue: str | None = None
        self.completed = 0
        self.results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ state
    @property
    def finished(self) -> bool:
        return self.completed >= self.n_trials

    @property
    def current(self) -> str | None:
        """Label of the cue currently on screen (None during gaps)."""
        return self._trial.cue if self._trial else None

    def progress(self, t: float) -> float | None:
        """Fraction (0..1) of the current cue window elapsed, for a HUD progress bar."""
        if not self._trial:
            return None
        return min(1.0, max(0.0, (t - self._trial.t_on) / self._trial.window))

    # ------------------------------------------------------------------ main
    def update(self, t: float, actions: Iterable[str], live: bool = True) -> list[dict[str, Any]]:
        """Advance the protocol; returns rows for any cues that completed this frame.

        While not ``live`` (paused / calibrating) the current cue is discarded and the
        next one is re-scheduled ``lead_in`` seconds after going live again.
        """
        acts = set(actions)
        done: list[dict[str, Any]] = []
        if not live or self.finished:
            self._trial = None
            self._next_start = None
            self._prev = acts
            return done
        onsets = acts - self._prev
        self._prev = acts
        if self._next_start is None:
            self._next_start = t + self.lead_in

        if self._trial is None:
            if t >= self._next_start:
                self._start(t)
            return done        # an onset in the same frame as the cue can't be a reaction

        tr = self._trial
        if onsets:
            if tr.expected is None:
                tr.false_activations += len(onsets)
                if tr.first_action is None:
                    tr.first_action, tr.first_action_t = sorted(onsets)[0], t
            else:
                if tr.first_action is None:
                    tr.first_action = tr.expected if tr.expected in onsets else sorted(onsets)[0]
                    tr.first_action_t = t
                if tr.expected in onsets and tr.match_t is None:
                    tr.match_t = t
                tr.wrong.extend(sorted(a for a in onsets if a != tr.expected))
        if t >= tr.t_on + tr.window:
            row = tr.row()
            done.append(row)
            self.results.append(row)
            self.completed += 1
            self._trial = None
            self._next_start = t + self.gap
        return done

    def _start(self, t: float) -> None:
        if not self._queue:
            block = [c for c in self.cues if self.cues[c] is not None]
            rests = [c for c in self.cues if self.cues[c] is None]
            if rests:
                block += [rests[0]] * self.rest_per_block
            self.rng.shuffle(block)
            if block and block[0] == self._last_cue and len(block) > 1:   # no immediate repeats
                block[0], block[-1] = block[-1], block[0]
            self._queue = block
        label = self._queue.pop(0)
        expected = self.cues[label]
        lo, hi = self.interval if expected is not None else self.rest_interval
        self._trial = _Trial(trial=self.completed + 1, cue=label, expected=expected,
                             t_on=t, window=self.rng.uniform(lo, hi))
        self._last_cue = label
        if self.on_cue is not None:
            self.on_cue(label, t)

