# Minecraft Biogaming — face + hand controller (SPEC)

Course: EN.580.471 Project 1 "Biogaming". Game Day Oct 30 2026. Target: Minecraft
**Education Edition** on **macOS (Apple Silicon)**. Must at minimum: move forward,
backward, left, right, jump, interact/right-click. Also wanted: look (camera),
attack/mine (left click), pause/kill switch.

Python ≥3.10 (user has Anaconda). Runs live from a webcam at ~30 fps.

## Hard constraints
- **mediapipe ≥ 0.10.30 Tasks API only** (`mediapipe.tasks.python.vision.HandLandmarker`,
  `FaceLandmarker`). The legacy `mp.solutions` API no longer exists. Model files
  (`hand_landmarker.task`, `face_landmarker.task`) are downloaded by
  `scripts/download_models.py` into `models/` on the user's machine (URLs:
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`).
  The build container CANNOT download them → nothing in tests may require them.
- Everything below the tracker must be testable with synthetic data (no camera, no models,
  no real key presses). Use dependency injection / a `DryRunOutput`.
- On any exit path (Ctrl-C, exception, pause) **all held keys and mouse buttons are released**.
- No network at runtime.

## Package layout (`mcbio/`)
```
mcbio/
  __init__.py
  types.py          # dataclasses shared by everyone (below) — owned by integrator, DO NOT change signatures
  filters.py        # OneEuroFilter (scalar + vector), EMA                      [Agent A]
  tracking.py       # Camera, HandTracker, FaceTracker (mediapipe Tasks, VIDEO mode)  [Agent A]
  features.py       # landmarks -> Features (head pose, blendshapes, palm pos, finger curl, pinch) [Agent A]
  gestures.py       # Rest/Flex/Hold/Extend curl state machine + debounced binary detectors [Agent B]
  mapping.py        # Features + Calibration + Profile -> Commands               [Agent B]
  calibration.py    # neutral pose capture, save/load JSON                       [Agent B]
  outputs.py        # OutputBackend: PynputOutput (mac), SerialLeonardoOutput, DryRunOutput [Agent B]
  hud.py            # OpenCV overlay window: landmarks, joystick box, active commands [Agent C]
  app.py            # main loop + CLI (python -m mcbio ...)                      [Agent C]
  logger.py         # CSV event/latency logging for experiments                  [Agent C]
profiles/*.yaml     # face_hand.yaml (default), face_only.yaml, hand_only.yaml  [Agent B]
arduino/leonardo_hid/leonardo_hid.ino   # serial -> Keyboard/Mouse HID          [Agent B]
scripts/download_models.py, scripts/analyze_session.py                         [Agent C]
tests/   (pytest, synthetic only)                                               [each agent its own]
README.md, requirements.txt                                                     [Agent C]
```

## Shared types (`mcbio/types.py`) — the contract
```python
@dataclass
class HeadPose:
    yaw: float      # degrees, + = user turns head to THEIR right (= screen-right in the mirrored view) -> camera turns right
    pitch: float    # degrees, + = looking up
    roll: float     # degrees, + = tilt head toward user's right shoulder

@dataclass
class HandFeatures:
    present: bool
    handedness: str                 # "Left"/"Right" as the USER sees it (correct for mirror)
    palm: tuple[float, float]       # palm centre in MIRRORED image coords, x,y in [0,1] (x right, y down)
    curl: tuple[float, float, float, float, float]   # thumb..little, 0 = straight, 1 = fully curled
    pinch: float                    # thumb-tip to index-tip distance / palm size (≈0 touching, ~1+ apart)
    scale: float                    # palm size in image units (wrist->middle MCP), for depth/normalising
    landmarks: list | None = None   # 21 x (x,y,z) mirrored normalized, for HUD

@dataclass
class FaceFeatures:
    present: bool
    head: HeadPose
    blend: dict[str, float]         # mediapipe blendshape name -> score 0..1 (jawOpen, browInnerUp, mouthSmileLeft, eyeBlinkLeft, ...)
    nose: tuple[float, float]       # nose tip, mirrored normalized image coords
    scale: float                    # face size (inter-ocular distance, image units)
    landmarks: list | None = None   # optional subset for HUD

@dataclass
class Features:
    t: float                        # seconds (monotonic) of the frame
    face: FaceFeatures
    hands: list[HandFeatures]       # 0..2

@dataclass
class Commands:
    keys: set[str]                  # held keys this frame, minecraft names: "forward","back","left","right","jump","sneak","sprint"
    mouse_dx: float                 # pixels to move this frame
    mouse_dy: float
    left_button: bool               # held
    right_button: bool              # held
    taps: list[str]                 # one-shot actions this frame e.g. "inventory","hotbar_next","hotbar_prev"
    paused: bool
    debug: dict                     # anything for HUD
```
Key bindings (default Minecraft): forward=W, back=S, left=A, right=D, jump=Space, sneak=Left Shift,
sprint=Left Ctrl, inventory=E, hotbar_next/prev = mouse scroll. Bindings live in the profile YAML.

## Default control scheme (profiles/face_hand.yaml)
- **Look:** head yaw → mouse_dx, head pitch → mouse_dy. Relative to calibrated neutral. Deadzone ±6°,
  rate control (velocity ∝ angle beyond deadzone, gain, cap), smoothed (OneEuro). This is "joystick" style
  so the user can keep looking at the screen.
- **Move:** dominant hand palm position relative to calibrated neutral box → forward/back (y) and
  left/right strafe (x). Deadzone radius, hysteresis (enter at r_in, exit at r_out<r_in) so keys don't chatter.
  Hand absent → release all movement.
- **Attack/mine (left click, held):** fist (mean curl of 4 fingers ≥ θ_on, release below θ_off) using the
  Rest/Flex/Hold/Extend curl state machine (reused idea from user's `dual/src/states.py`).
- **Interact/place (right click):** thumb–index pinch (pinch < p_on, release > p_off), min hold 80 ms.
- **Jump:** mouth open (blend jawOpen ≥ j_on, release < j_off). Held while open (so you can hold to swim/climb).
- **Sneak:** optional, eyebrows raised (browInnerUp).
- **Pause/resume:** both hands open palms shown for 1.0 s, OR keyboard hotkey (F8) — paused releases everything.
- **Kill switch:** Esc in HUD window or Ctrl-C.

### face_only.yaml (no-hands users: e.g. bilateral transradial amputees, SCI)
- Move: head roll left/right → strafe; nose position relative to neutral (forward lean = face scale up, or nose y up) →
  forward/back. Look: yaw/pitch as above but with a "look mode" toggle (smile) vs "walk mode" so look and move don't fight.
- Jump: mouth open. Left click: smile-left? keep simple & reliable: **brow raise = left click (attack)**,
  **pucker/mouthPucker or cheekPuff = right click (interact)**. Toggle look/walk: long blink both eyes 0.6 s.
  Exact choices configurable; implement generically.

### hand_only.yaml — movement + clicks from hand, look from index-finger pointing direction / second hand.

## Mapping must be generic
Profile YAML defines a list of *bindings*: `{action: forward, source: hand.palm_y, op: "<", on: -0.08, off: -0.05}`,
`{action: left_button, source: hand.fist, ...}`, `{action: jump, source: face.blend.jawOpen, op: ">", on: 0.45, off: 0.3, min_hold_ms: 0}`,
plus `look: {yaw_source, pitch_source, deadzone_deg, gain, max_px_per_frame, invert_y}`.
Sources are dotted paths resolved from a flat dict produced by `features.flatten(Features, Calibration) -> dict[str,float]`
(owned by Agent A; `Calibration` is in types.py). It returns CONTINUOUS calibrated, signed values only:
- `face.present` (0/1), `face.yaw`, `face.pitch`, `face.roll` (degrees minus neutral),
- `face.blend.<name>` (score minus neutral_blend, clipped ≥0), `face.nose_dx`, `face.nose_dy` (minus neutral, in units of face scale),
  `face.scale_ratio` (face scale / neutral − 1; + = leaning toward camera),
- `hand.present` (0/1, dominant hand), `hand.palm_dx`, `hand.palm_dy` (minus neutral, divided by palm_range, + = right / + = UP i.e. y flipped),
  `hand.curl_mean` (mean curl index..little), `hand.curl_<finger>`, `hand.pinch`,
- `offhand.present`, `offhand.curl_mean`, `offhand.pinch`, `hands.count`.
Missing face/hand → its keys are present with value 0 and `*.present` = 0.
Binary gestures (fist, pinch, open palm, mouth open, both palms open for pause…) are derived from these
continuous values by `gestures.py` (Agent B) using hysteresis + min-hold debouncing and the
Rest/Flex/Hold/Extend curl state machine.

## Outputs
- `PynputOutput`: pynput keyboard + mouse. On macOS relative mouse moves for Minecraft camera must be sent as
  *relative* events; implement with Quartz `CGEventCreateMouseEvent` + `kCGMouseEventDeltaX/Y` when `Quartz`
  importable (pyobjc), else pynput `mouse.move(dx,dy)`. Document macOS Accessibility + Input Monitoring + Camera permissions.
- `SerialLeonardoOutput`: sends compact lines over USB serial at 115200: `K+w` / `K-w` key down/up,
  `M dx dy` relative move (clamped −127..127, split if larger), `B+L`/`B-L`/`B+R`/`B-R`, `R` release all,
  `P` ping→`OK`. Only sends *changes* (edge-triggered) + mouse moves. Arduino sketch implements it with
  Keyboard.h / Mouse.h and releases all if no message for 500 ms (watchdog).
- `DryRunOutput`: records calls (for tests/HUD).
All backends implement: `apply(cmds: Commands)`, `release_all()`, `close()`. Edge-triggered: press on rising, release on falling.

## Experiment / grad requirement
`logger.py` writes per-frame CSV (t, latency_ms capture→output, active actions, key features) and an events CSV
(action on/off with timestamps). `scripts/analyze_session.py` → summary table + plots (latency hist, actions/min,
false activations during a scripted "rest" period). A `--protocol` mode in app shows on-screen cues
("JUMP", "FORWARD", …) at random intervals and logs cue→action reaction time & accuracy, so the team can test 5 people.

## Quality bar
- `python -m pytest -q` passes in the container with no camera/models.
- `python -m mcbio --dry-run --synthetic` runs headless for N frames using a synthetic feature generator (no mediapipe)
  and prints commands — proves the whole pipeline end to end.
- Clean, commented code; README explains setup on macOS step by step.
