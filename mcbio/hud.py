"""Heads-up display: an OpenCV overlay drawn on the (mirrored) camera frame.

Everything here is a *pure drawing function* that takes a BGR ``numpy`` frame and returns
a frame of the same shape, so it is testable headless (``opencv-python-headless`` has no
``imshow``). Only :class:`HudWindow` touches the window system.

Two views, toggled with **H** in the camera window:

* **simple** (default): status pill + fps, chips for only the actions firing right now
  (taps such as INVENTORY / SLOT > flash amber), the hand skeleton and joystick centre,
  and small MOVE / LOOK gauges. No face mesh, no tuning bars.
* **detailed** (for tuning): the full layout below.

Detailed layout (designed to be readable on a projector and in a recorded pitch video)::

    +---------------------------------------------------------------+
    | [LIVE] profile/backend                       30 fps   18 ms   |  status bar
    | [W] [A] [S] [D] [JUMP] [SNEAK] [SPRINT] [L-CLICK] [R-CLICK] <> |  action chips
    |                     ( protocol cue / banners )                |
    |                 hand skeleton + face points                   |
    | +--MOVE--+ +--LOOK--+                    +-- HAND / FACE ---+ |
    | | stick  | | gauge  |                    | Rest Flex Hold Ex| |
    | +--------+ +--------+                    | curl/pinch/jaw   | |
    +---------------------------------------------------------------+
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

try:  # cv2 is a hard dependency at runtime, but keep import errors readable
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("hud.py needs OpenCV: pip install opencv-python") from e

from mcbio.logger import active_actions
from mcbio.types import Commands, Features

# ---------------------------------------------------------------------- palette (BGR)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
DARK = (30, 30, 30)
GREEN = (90, 210, 90)
AMBER = (0, 175, 255)
RED = (60, 60, 230)
BLUE = (230, 160, 60)
CYAN = (230, 220, 80)
LIT = (70, 225, 120)         # lit chip
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # little
    (0, 17),
)

#: (chip label, logical action) in display order.
CHIPS: tuple[tuple[str, str], ...] = (
    ("W", "forward"), ("A", "left"), ("S", "back"), ("D", "right"),
    ("JUMP", "jump"), ("SNEAK", "sneak"), ("SPRINT", "sprint"),
    ("L-CLICK", "attack"), ("R-CLICK", "interact"),
)
CURL_STATES: tuple[str, ...] = ("Rest", "Flex", "Hold", "Extend")

#: Simple HUD: (logical action or tap, chip text) in display order. Only active ones are drawn.
SIMPLE_LABELS: tuple[tuple[str, str], ...] = (
    ("forward", "W"), ("left", "A"), ("back", "S"), ("right", "D"),
    ("jump", "JUMP"), ("sprint", "SPRINT"), ("sneak", "SNEAK"),
    ("attack", "MINE"), ("interact", "PLACE"),
    ("look_left", "LOOK <"), ("look_right", "LOOK >"), ("look_up", "LOOK ^"), ("look_down", "LOOK v"),
    ("inventory", "INVENTORY"), ("hotbar_prev", "< SLOT"), ("hotbar_next", "SLOT >"),
)


@dataclass
class HudState:
    """Everything the HUD shows that isn't in Features/Commands.

    ``status`` is one of LIVE, PAUSED, ARMING (countdown before keys go live), CALIBRATING.
    """
    status: str = "LIVE"
    fps: float = 0.0
    latency_ms: float = 0.0
    countdown: float | None = None            # seconds left while ARMING
    prompt: tuple[str, str] | None = None     # calibration (title, subtitle)
    prompt_progress: float | None = None      # 0..1 capture progress
    prompt_count: int | None = None           # big "3, 2, 1" number
    cue: str | None = None                    # protocol cue on screen
    cue_progress: float | None = None
    cue_info: str | None = None               # e.g. "trial 4/30"
    profile: str = ""
    backend: str = ""
    message: str | None = None                # small toast at the bottom centre
    hotkey_ok: bool = True
    pause_hint: str = "show both open palms"     # profile's pause gesture, for the banner
    actions: set[str] | None = None           # override active actions (else derived)
    neutral_palm: tuple[float, float] | None = None   # calibrated palm neutral (image)
    detail: bool = False                      # False = simple HUD, True = full tuning view (H key)
    recent_taps: list[str] = field(default_factory=list)   # taps fired in the last ~0.8 s
    palm_range: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ====================================================================== primitives
def _s(frame: np.ndarray) -> float:
    """UI scale factor: 1.0 at 1280 px wide (min 0.5 so tiny test frames still work)."""
    return max(0.5, frame.shape[1] / 1280.0)


def panel(frame: np.ndarray, x: int, y: int, w: int, h: int,
          color: tuple[int, int, int] = DARK, alpha: float = 0.65) -> np.ndarray:
    """Alpha-blend a filled rectangle onto ``frame`` in place (clipped to the frame)."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return frame
    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)
    return frame


def text(frame: np.ndarray, s: str, org: tuple[int, int], scale: float = 1.0,
         color: tuple[int, int, int] = WHITE, thick: int = 2, font: int = FONT,
         center: bool = False, outline: bool = True) -> tuple[int, int]:
    """Draw text with a dark outline for legibility; returns (width, height)."""
    (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
    x, y = org
    if center:
        x -= tw // 2
    if outline:
        cv2.putText(frame, s, (x, y), font, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return tw, th


def wrap(s: str, max_w: int, scale: float, thick: int = 2, font: int = FONT) -> list[str]:
    """Greedy word wrap to a pixel width."""
    lines: list[str] = []
    cur = ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if cv2.getTextSize(trial, font, scale, thick)[0][0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _px(pt: Sequence[float], W: int, H: int) -> tuple[int, int]:
    return int(round(pt[0] * W)), int(round(pt[1] * H))


def _bar(frame: np.ndarray, x: int, y: int, w: int, h: int, value: float,
         color: tuple[int, int, int], marks: Iterable[float] = ()) -> None:
    """Horizontal 0..1 bar with optional threshold tick marks."""
    v = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), GREY, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x + 1, y + 1), (x + 1 + int((w - 2) * v), y + h - 1), color, -1)
    for m in marks:
        mx = x + int(w * float(np.clip(m, 0, 1)))
        cv2.line(frame, (mx, y - 3), (mx, y + h + 3), WHITE, 2, cv2.LINE_AA)


def _dbg(debug: dict | None, *keys: str, default: Any = None) -> Any:
    """First present key among ``keys`` in ``debug`` (tolerant to naming differences)."""
    if not debug:
        return default
    for k in keys:
        if k in debug and debug[k] is not None:
            return debug[k]
    return default


# ====================================================================== landmarks
def draw_hand(frame: np.ndarray, landmarks: Sequence[Sequence[float]] | None,
              color: tuple[int, int, int] = GREEN) -> np.ndarray:
    """Draw a 21-point hand skeleton (mirrored normalized coords)."""
    if not landmarks or len(landmarks) < 21:
        return frame
    H, W = frame.shape[:2]
    s = _s(frame)
    pts = [_px(p, W, H) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, max(2, int(3 * s)), cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = int((7 if i in (4, 8, 12, 16, 20) else 5) * s)
        cv2.circle(frame, p, r, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, p, r, color, 2, cv2.LINE_AA)
    return frame


def draw_face(frame: np.ndarray, features: Features) -> np.ndarray:
    """Draw face key points (or just the nose tip if no landmark subset is given)."""
    face = features.face
    if not face.present:
        return frame
    H, W = frame.shape[:2]
    s = _s(frame)
    lms = face.landmarks or []
    if len(lms) > 120:                    # full mesh: subsample to keep it readable
        lms = lms[::8]
    for p in lms:
        cv2.circle(frame, _px(p, W, H), max(2, int(3 * s)), CYAN, -1, cv2.LINE_AA)
    cv2.circle(frame, _px(face.nose, W, H), int(8 * s), AMBER, -1, cv2.LINE_AA)
    return frame


def draw_neutral_marker(frame: np.ndarray, neutral: tuple[float, float] | None,
                        palm_range: float | None, palm: tuple[float, float] | None,
                        deadzone: float) -> np.ndarray:
    """Show the calibrated neutral palm spot + deadzone ring on the camera image."""
    if neutral is None or not palm_range:
        return frame
    H, W = frame.shape[:2]
    c = _px(neutral, W, H)
    r = max(4, int(deadzone * palm_range * W))
    cv2.circle(frame, c, r, WHITE, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, c, WHITE, cv2.MARKER_CROSS, int(18 * _s(frame)), 2)
    if palm is not None:
        cv2.line(frame, c, _px(palm, W, H), AMBER, 2, cv2.LINE_AA)
    return frame


# ====================================================================== widgets
def draw_status_bar(frame: np.ndarray, state: HudState) -> np.ndarray:
    """Top bar: status pill, profile/backend, fps and latency."""
    H, W = frame.shape[:2]
    s = _s(frame)
    h = int(48 * s)
    panel(frame, 0, 0, W, h, BLACK, 0.6)
    col = {"LIVE": GREEN, "PAUSED": RED, "ARMING": AMBER, "CALIBRATING": BLUE}.get(state.status, GREY)
    label = state.status if state.status != "ARMING" else "STARTING"
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.9 * s, 2)
    cv2.rectangle(frame, (int(10 * s), int(8 * s)), (int(10 * s) + tw + int(20 * s), h - int(8 * s)), col, -1)
    cv2.putText(frame, label, (int(20 * s), h - int(15 * s)), FONT_BOLD, 0.9 * s, BLACK, 2, cv2.LINE_AA)
    info = " / ".join(x for x in (state.profile, state.backend) if x)
    if not state.hotkey_ok:
        info += "   (F8 hotkey off: press P in this window)"
    text(frame, info, (tw + int(45 * s), h - int(16 * s)), 0.6 * s, WHITE, 1)
    perf = f"{state.fps:4.0f} fps   {state.latency_ms:5.1f} ms"
    (pw, _), _ = cv2.getTextSize(perf, FONT, 0.8 * s, 2)
    lat_col = GREEN if state.latency_ms < 50 else (AMBER if state.latency_ms < 100 else RED)
    text(frame, perf, (W - pw - int(15 * s), h - int(15 * s)), 0.8 * s, lat_col, 2)
    return frame


def draw_status_simple(frame: np.ndarray, state: HudState) -> np.ndarray:
    """Simple HUD top line: status pill + fps, and a quiet "H: details" hint on the right."""
    H, W = frame.shape[:2]
    s = _s(frame)
    col = {"LIVE": GREEN, "PAUSED": RED, "ARMING": AMBER, "CALIBRATING": BLUE}.get(state.status, GREY)
    label = state.status if state.status != "ARMING" else "STARTING"
    x0, y0 = int(10 * s), int(10 * s)
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.8 * s, 2)
    h = th + int(18 * s)
    cv2.rectangle(frame, (x0, y0), (x0 + tw + int(24 * s), y0 + h), col, -1)
    cv2.putText(frame, label, (x0 + int(12 * s), y0 + h - int(9 * s)), FONT_BOLD, 0.8 * s, BLACK, 2, cv2.LINE_AA)
    text(frame, f"{state.fps:.0f} fps", (x0 + tw + int(36 * s), y0 + h - int(9 * s)), 0.55 * s, GREY, 1)
    hint = "H: details"
    (hw, _), _ = cv2.getTextSize(hint, FONT, 0.5 * s, 1)
    text(frame, hint, (W - hw - int(12 * s), y0 + h - int(9 * s)), 0.5 * s, GREY, 1)
    return frame


def draw_active_chips(frame: np.ndarray, actions: Iterable[str], taps: Iterable[str] = ()) -> np.ndarray:
    """Simple HUD: only what is firing right now (green), plus recent one-shot taps (amber)."""
    H, W = frame.shape[:2]
    s = _s(frame)
    tap_set = set(taps)
    live = set(actions) | tap_set
    x, y, h = int(10 * s), int(56 * s), int(40 * s)
    pad = int(12 * s)
    for act, label in SIMPLE_LABELS:
        if act not in live:
            continue
        (tw, _), _ = cv2.getTextSize(label, FONT_BOLD, 0.75 * s, 2)
        w = tw + 2 * pad
        if x + w > W - 5:
            break
        cv2.rectangle(frame, (x, y), (x + w, y + h), AMBER if act in tap_set else LIT, -1)
        cv2.putText(frame, label, (x + pad, y + h - int(12 * s)), FONT_BOLD, 0.75 * s, BLACK, 2, cv2.LINE_AA)
        x += w + int(6 * s)
    return frame


def draw_action_chips(frame: np.ndarray, actions: set[str], taps: Sequence[str] = ()) -> np.ndarray:
    """Row of chips (W A S D JUMP SNEAK SPRINT L-CLICK R-CLICK + look arrows); lit = active."""
    H, W = frame.shape[:2]
    s = _s(frame)
    x, y, h = int(10 * s), int(58 * s), int(44 * s)
    pad = int(14 * s)
    items = list(CHIPS) + [("<", "look_left"), (">", "look_right"), ("^", "look_up"), ("v", "look_down")]
    for label, act in items:
        (tw, _), _ = cv2.getTextSize(label, FONT_BOLD, 0.8 * s, 2)
        w = tw + 2 * pad
        if x + w > W - 5:
            break
        on = act in actions
        if on:
            cv2.rectangle(frame, (x, y), (x + w, y + h), LIT, -1)
        else:
            panel(frame, x, y, w, h, DARK, 0.6)
        cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE if on else GREY, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x + pad, y + h - int(13 * s)), FONT_BOLD, 0.8 * s,
                    BLACK if on else (200, 200, 200), 2, cv2.LINE_AA)
        x += w + int(6 * s)
    if taps:
        text(frame, "tap: " + " ".join(taps), (int(12 * s), y + h + int(28 * s)), 0.7 * s, AMBER, 2)
    return frame


def draw_joystick(frame: np.ndarray, origin: tuple[int, int], size: int,
                  dx: float, dy: float, present: bool, deadzone: float = 0.35,
                  deadzone_out: float | None = None, full: float = 1.5,
                  active: set[str] | None = None) -> np.ndarray:
    """Palm "joystick": neutral cross, deadzone ring(s) and the current palm position.

    ``dx``/``dy`` are calibrated palm offsets in units of palm_range (+dy = up). ``full``
    is the value at the box edge. The inner ring is the *enter* threshold, the faint
    outer/inner ring the *exit* threshold (hysteresis).
    """
    x, y = origin
    s = _s(frame)
    panel(frame, x, y, size, size + int(34 * s), DARK, 0.7)
    text(frame, "MOVE (hand)", (x + int(8 * s), y + int(24 * s)), 0.6 * s, WHITE, 1)
    top = y + int(34 * s)
    cx, cy = x + size // 2, top + size // 2
    half = size // 2 - int(10 * s)
    k = half / full
    cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), GREY, 1, cv2.LINE_AA)
    cv2.line(frame, (cx - half, cy), (cx + half, cy), (80, 80, 80), 1)
    cv2.line(frame, (cx, cy - half), (cx, cy + half), (80, 80, 80), 1)
    # direction labels light up with the active key
    act = active or set()
    for lab, a, (lx, ly) in (("W", "forward", (cx, cy - half + int(18 * s))),
                             ("S", "back", (cx, cy + half - int(6 * s))),
                             ("A", "left", (cx - half + int(12 * s), cy + int(6 * s))),
                             ("D", "right", (cx + half - int(12 * s), cy + int(6 * s)))):
        text(frame, lab, (lx, ly), 0.6 * s, LIT if a in act else GREY, 2, center=True)
    cv2.circle(frame, (cx, cy), max(2, int(deadzone * k)), WHITE, 2, cv2.LINE_AA)
    if deadzone_out is not None and deadzone_out != deadzone:
        cv2.circle(frame, (cx, cy), max(2, int(deadzone_out * k)), GREY, 1, cv2.LINE_AA)
    if not present:
        text(frame, "NO HAND", (cx, cy + int(5 * s)), 0.6 * s, RED, 2, center=True)
        return frame
    px = int(cx + np.clip(dx, -full, full) * k)
    py = int(cy - np.clip(dy, -full, full) * k)
    outside = (dx * dx + dy * dy) ** 0.5 > deadzone
    cv2.line(frame, (cx, cy), (px, py), AMBER, 2, cv2.LINE_AA)
    cv2.circle(frame, (px, py), int(10 * s), LIT if outside else WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, (px, py), int(10 * s), BLACK, 2, cv2.LINE_AA)
    return frame


def draw_head_gauge(frame: np.ndarray, origin: tuple[int, int], size: int,
                    yaw: float, pitch: float, present: bool, deadzone_deg: float = 6.0,
                    full_deg: float = 30.0, mouse: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Head yaw/pitch gauge with the deadzone square (rate-control "look stick")."""
    x, y = origin
    s = _s(frame)
    panel(frame, x, y, size, size + int(34 * s), DARK, 0.7)
    text(frame, "LOOK (head)", (x + int(8 * s), y + int(24 * s)), 0.6 * s, WHITE, 1)
    top = y + int(34 * s)
    cx, cy = x + size // 2, top + size // 2
    half = size // 2 - int(10 * s)
    k = half / full_deg
    cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), GREY, 1, cv2.LINE_AA)
    d = max(2, int(deadzone_deg * k))
    cv2.rectangle(frame, (cx - d, cy - d), (cx + d, cy + d), WHITE, 2, cv2.LINE_AA)
    if not present:
        text(frame, "NO FACE", (cx, cy + int(5 * s)), 0.6 * s, RED, 2, center=True)
        return frame
    px = int(cx + np.clip(yaw, -full_deg, full_deg) * k)
    py = int(cy - np.clip(pitch, -full_deg, full_deg) * k)
    outside = abs(yaw) > deadzone_deg or abs(pitch) > deadzone_deg
    cv2.line(frame, (cx, cy), (px, py), AMBER, 2, cv2.LINE_AA)
    cv2.circle(frame, (px, py), int(10 * s), LIT if outside else WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, (px, py), int(10 * s), BLACK, 2, cv2.LINE_AA)
    text(frame, f"yaw {yaw:+.0f}  pitch {pitch:+.0f}", (x + int(8 * s), top + size - int(4 * s)),
         0.5 * s, WHITE, 1)
    if mouse != (0.0, 0.0):
        text(frame, f"mouse {mouse[0]:+.0f},{mouse[1]:+.0f}", (x + int(8 * s), top + int(18 * s)),
             0.5 * s, AMBER, 1)
    return frame


def draw_gesture_panel(frame: np.ndarray, origin: tuple[int, int], width: int,
                       curl_state: str | None, curl_mean: float, pinch: float,
                       jaw: float, brow: float, thresholds: dict[str, float] | None = None) -> np.ndarray:
    """Curl state machine (Rest/Flex/Hold/Extend) + curl, pinch, jaw, brow level bars."""
    x, y = origin
    s = _s(frame)
    th = thresholds or {}
    row = int(34 * s)
    h = int(40 * s) + 5 * row
    panel(frame, x, y, width, h, DARK, 0.7)
    text(frame, "HAND / FACE", (x + int(8 * s), y + int(24 * s)), 0.6 * s, WHITE, 1)
    # state chips
    cy = y + int(36 * s)
    cw = (width - int(16 * s)) // 4
    cur = (curl_state or "").strip().lower()
    for i, name in enumerate(CURL_STATES):
        cx = x + int(8 * s) + i * cw
        on = cur == name.lower()
        if on:
            cv2.rectangle(frame, (cx, cy), (cx + cw - int(4 * s), cy + row - int(4 * s)), BLUE, -1)
        cv2.rectangle(frame, (cx, cy), (cx + cw - int(4 * s), cy + row - int(4 * s)), WHITE if on else GREY, 1)
        text(frame, name, (cx + (cw - int(4 * s)) // 2, cy + row - int(13 * s)), 0.55 * s,
             BLACK if on else (200, 200, 200), 2 if on else 1, center=True, outline=False)
    bars = (
        ("curl", curl_mean, GREEN, [th.get("fist_on", 0.7), th.get("fist_off", 0.5)]),
        ("pinch", 1.0 - min(1.0, pinch), AMBER, [1.0 - th.get("pinch_on", 0.25), 1.0 - th.get("pinch_off", 0.4)]),
        ("jaw", jaw, CYAN, [th.get("jaw_on", 0.45), th.get("jaw_off", 0.3)]),
        ("brow", brow, BLUE, [th.get("brow_on", 0.5), th.get("brow_off", 0.35)]),
    )
    bx = x + int(80 * s)
    bw = width - int(95 * s)
    for i, (name, v, col, marks) in enumerate(bars):
        by = cy + row * (i + 1) + int(6 * s)
        text(frame, name, (x + int(8 * s), by + int(16 * s)), 0.55 * s, WHITE, 1)
        _bar(frame, bx, by + int(2 * s), bw, int(16 * s), v, col, marks)
    return frame


def draw_banner(frame: np.ndarray, title: str, subtitle: str = "",
                color: tuple[int, int, int] = RED) -> np.ndarray:
    """Big centred banner (PAUSED / STARTING IN 3)."""
    H, W = frame.shape[:2]
    s = _s(frame)
    bh = int(170 * s)
    y = H // 2 - bh // 2
    panel(frame, 0, y, W, bh, color, 0.55)
    text(frame, title, (W // 2, y + int(95 * s)), 2.6 * s, WHITE, max(3, int(5 * s)), FONT_BOLD, center=True)
    if subtitle:
        text(frame, subtitle, (W // 2, y + int(145 * s)), 0.85 * s, WHITE, 2, center=True)
    return frame


def draw_prompt(frame: np.ndarray, title: str, subtitle: str = "",
                progress: float | None = None, count: int | None = None) -> np.ndarray:
    """Calibration prompt: dims the frame, big instructions, optional 3-2-1 and progress."""
    H, W = frame.shape[:2]
    s = _s(frame)
    panel(frame, 0, 0, W, H, BLACK, 0.45)
    y = int(H * 0.30)
    for line in wrap(title, int(W * 0.9), 1.4 * s, 3, FONT_BOLD):
        text(frame, line, (W // 2, y), 1.4 * s, WHITE, 3, FONT_BOLD, center=True)
        y += int(55 * s)
    for line in wrap(subtitle, int(W * 0.85), 0.9 * s, 2):
        text(frame, line, (W // 2, y), 0.9 * s, (220, 220, 220), 2, center=True)
        y += int(40 * s)
    if count is not None:
        text(frame, str(count), (W // 2, y + int(110 * s)), 4.0 * s, AMBER, 8, FONT_BOLD, center=True)
    if progress is not None:
        bw, bh = int(W * 0.5), int(22 * s)
        _bar(frame, (W - bw) // 2, int(H * 0.82), bw, bh, progress, BLUE)
    return frame


def draw_cue(frame: np.ndarray, cue: str, progress: float | None = None,
             info: str | None = None) -> np.ndarray:
    """Protocol cue in big letters near the top centre."""
    H, W = frame.shape[:2]
    s = _s(frame)
    y0 = int(120 * s)
    col = GREY if cue == "REST" else (40, 120, 40)
    (tw, th), _ = cv2.getTextSize(cue, FONT_BOLD, 2.4 * s, 5)
    bw = tw + int(60 * s)
    panel(frame, W // 2 - bw // 2, y0, bw, th + int(50 * s), col, 0.75)
    text(frame, cue, (W // 2, y0 + th + int(22 * s)), 2.4 * s, WHITE, 5, FONT_BOLD, center=True)
    if progress is not None:
        _bar(frame, W // 2 - bw // 2, y0 + th + int(50 * s), bw, int(10 * s), 1.0 - progress, WHITE)
    if info:
        text(frame, info, (W // 2, y0 + th + int(85 * s)), 0.6 * s, WHITE, 1, center=True)
    return frame


# ====================================================================== composite
def draw_hud(frame: np.ndarray, features: Features | None, commands: Commands | None,
             debug: dict | None = None, flat: dict[str, float] | None = None,
             state: HudState | None = None, mirror: bool = False) -> np.ndarray:
    """Draw the full HUD and return a new frame (the input is not modified).

    Args:
        frame: BGR image (H, W, 3) uint8. If ``mirror`` the frame is flipped
            horizontally first (raw webcam -> selfie view, matching landmark coords).
        features: latest tracker output (landmarks are drawn if present).
        commands: latest commands (active action chips, mouse motion).
        debug: mapper debug dict (defaults to ``commands.debug``). Recognised keys:
            ``curl_state``, ``move_deadzone``/``r_in``, ``move_deadzone_out``/``r_out``,
            ``look_deadzone_deg``/``deadzone_deg``, ``thresholds``.
        flat: flattened calibrated features (``hand.palm_dx`` etc.) for the gauges.
        state: status/fps/latency/banners/prompts.
    """
    out = cv2.flip(frame, 1) if mirror else frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    st = state or HudState()
    cmds = commands or Commands.idle()
    dbg = debug if debug is not None else cmds.debug
    fl = flat or {}
    H, W = out.shape[:2]
    s = _s(out)

    dz = float(_dbg(dbg, "move_deadzone", "r_in", "move_r_in", default=0.35))
    dz_out = _dbg(dbg, "move_deadzone_out", "r_out", "move_r_out")
    look_dz = float(_dbg(dbg, "look_deadzone_deg", "deadzone_deg", default=6.0))

    # --- camera-space overlays
    if features is not None:
        if st.detail:                     # face mesh dots only in the detailed view
            draw_face(out, features)
        for i, hand in enumerate(features.hands):
            if hand.present:
                draw_hand(out, hand.landmarks, GREEN if i == 0 else BLUE)
        dom = features.hands[0].palm if features.hands else None
        draw_neutral_marker(out, st.neutral_palm, st.palm_range, dom, dz)

    actions = st.actions if st.actions is not None else active_actions(cmds)
    if st.detail:
        draw_status_bar(out, st)
        draw_action_chips(out, actions, cmds.taps)
    else:
        draw_status_simple(out, st)
        draw_active_chips(out, actions, list(cmds.taps) + list(st.recent_taps))

    # --- bottom widgets (smaller in the simple view)
    size = int(min(W * 0.18, H * 0.30)) if st.detail else int(min(W * 0.13, H * 0.22))
    by = H - size - int(44 * s)
    hand_present = bool(fl.get("hand.present", 1.0 if (features and features.hands) else 0.0))
    face_present = bool(fl.get("face.present", 1.0 if (features and features.face.present) else 0.0))
    draw_joystick(out, (int(10 * s), by), size, float(fl.get("hand.palm_dx", 0.0)),
                  float(fl.get("hand.palm_dy", 0.0)), hand_present, dz,
                  float(dz_out) if dz_out is not None else None, active=actions)
    draw_head_gauge(out, (int(20 * s) + size, by), size, float(fl.get("face.yaw", 0.0)),
                    float(fl.get("face.pitch", 0.0)), face_present, look_dz,
                    mouse=(cmds.mouse_dx, cmds.mouse_dy))
    if st.detail:                         # tuning panel: curl state machine + threshold bars
        gw = int(min(W * 0.32, 420 * s))
        curl_state = _dbg(dbg, "curl_state", "fist_state", "curl")
        draw_gesture_panel(out, (W - gw - int(10 * s), by - int(10 * s)), gw,
                           str(curl_state) if curl_state is not None else None,
                           float(fl.get("hand.curl_mean", 0.0)), float(fl.get("hand.pinch", 1.0)),
                           float(fl.get("face.blend.jawOpen", 0.0)), float(fl.get("face.blend.browInnerUp", 0.0)),
                           _dbg(dbg, "thresholds") if isinstance(_dbg(dbg, "thresholds"), dict) else None)

    # --- centre overlays (highest priority last)
    if st.cue and st.status == "LIVE":
        draw_cue(out, st.cue, st.cue_progress, st.cue_info)
    if st.message:
        text(out, st.message, (W // 2, H - int(14 * s)), 0.7 * s, AMBER, 2, center=True)
    if st.status == "PAUSED":
        draw_banner(out, "PAUSED", f"F8 or {st.pause_hint} to resume   |   Esc to quit", RED)
    elif st.status == "ARMING":
        n = int(np.ceil(st.countdown or 0))
        draw_banner(out, f"STARTING IN {n}", "switch to Minecraft now  -  F8 to pause", (0, 120, 200))
    if st.status == "CALIBRATING" and st.prompt:
        draw_prompt(out, st.prompt[0], st.prompt[1], st.prompt_progress, st.prompt_count)
    return out


def blank_frame(width: int = 1280, height: int = 720, label: str = "SYNTHETIC") -> np.ndarray:
    """A dark gradient background (used when there is no camera image)."""
    g = np.linspace(40, 15, height, dtype=np.uint8)[:, None]
    f = np.repeat(np.repeat(g, width, axis=1)[:, :, None], 3, axis=2).copy()
    if label:
        s = width / 1280.0
        cv2.putText(f, label, (int(20 * s), height // 2), FONT_BOLD, 3.0 * s, (55, 55, 55), 6, cv2.LINE_AA)
    return f


# ====================================================================== window
def gui_unavailable_reason() -> str | None:
    """Why OpenCV can't open a window here (None = it probably can).

    Checked *before* calling ``cv2.namedWindow`` because on Linux without a display the
    Qt/GTK backends abort the whole process instead of raising.
    """
    import os
    import sys
    info = cv2.getBuildInformation()
    gui_line = next((ln for ln in info.splitlines() if ln.strip().startswith("GUI:")), "")
    if gui_line and gui_line.split(":", 1)[1].strip().upper() in ("NONE", "NO"):
        return "OpenCV built without GUI (opencv-python-headless?) - pip install opencv-python"
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no display (DISPLAY not set)"
    return None


class HudWindow:
    """Thin ``cv2.imshow`` wrapper; degrades to "no window" when the GUI is unavailable."""

    def __init__(self, title: str = "mcbio HUD", width: int | None = None) -> None:
        self.title = title
        self.width = width
        self.available = True
        self.error = ""
        reason = gui_unavailable_reason()
        if reason:
            self.available = False
            self.error = reason
            return
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            if width:
                cv2.resizeWindow(title, width, int(width * 9 / 16))
        except cv2.error as e:   # opencv-python-headless or no display
            self.available = False
            self.error = str(e).strip().splitlines()[-1] if str(e) else "no GUI"

    def show(self, frame: np.ndarray) -> int:
        """Show a frame; returns the key code pressed (or -1)."""
        if not self.available:
            return -1
        try:
            cv2.imshow(self.title, frame)
            key = cv2.waitKey(1)
            # window closed with the mouse -> behave like Esc
            if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                return 27
            return key
        except cv2.error:
            self.available = False
            return -1

    def close(self) -> None:
        if self.available:
            try:
                cv2.destroyWindow(self.title)
                cv2.waitKey(1)
            except cv2.error:
                pass
