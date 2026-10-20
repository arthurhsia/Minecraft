# Minecraft Biogaming: play Minecraft with your face and hands

**Turn your head to look. Move your hand to walk. Make a fist to mine, pinch to place, open your mouth to jump.**

`mcbio` is a webcam-only controller for **Minecraft Education Edition** on macOS. It needs no keyboard, mouse or special hardware: just a laptop camera. It was built for **EN.580.471/771 Biomedical Instrumentation – Project 1 "Biogaming"** (Johns Hopkins, Game Day Oct 30 2026), and is structured like any biomedical instrument:

```
 camera ──► MediaPipe landmarks ──► calibrated features ──► filters ──► hysteresis comparators ──► keyboard / mouse
(sensor)   (transducer + model)    (offset & gain)         (low-pass)   (decision)                  (actuator)
```

Three control profiles are included: **face + hand** (default), **face only** (for players who can't use their hands) and **hand only**.

---

## Contents
- [Features](#features)
- [Quick start (macOS)](#quick-start-macos)
- [Controls](#controls)
- [Calibration](#calibration)
- [Safety](#safety)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Command-line usage](#command-line-usage)
- [Tuning the controls](#tuning-the-controls)
- [Arduino Leonardo option](#arduino-leonardo-option-hardware-hid)
- [Experiment protocol](#experiment-protocol-5-subjects)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Mapping to course concepts](#mapping-to-course-concepts)
- [Acknowledgements](#acknowledgements)

---

## Features
- **Hands-free capable.** Every Minecraft action required by the project brief (forward, back, left, right, jump, interact) plus look, attack/mine, sprint, sneak and pause.
- **Runs live at ~30 fps** with sub-millisecond processing latency after tracking.
- **Personal calibration.** Everything is measured relative to *your* resting pose and *your* comfortable reach.
- **No flicker.** Every control is a Schmitt-trigger-style comparator (separate on/off thresholds) with optional debounce timers, and signals are smoothed with a One-Euro filter.
- **Safe by design.** Input is only sent while Minecraft is the front app, everything is released on pause, app switch, quit or crash, and it starts paused.
- **Fully configurable.** All gestures and thresholds live in plain YAML profiles; no code changes needed.
- **Live HUD.** The camera window shows tracking, a joystick box, a look gauge, the finger-curl state and which actions are firing.
- **Built-in experiment mode.** Cue-based reaction-time and accuracy testing with CSV logging and an analysis script (mean ± SD, false activations at rest, latency).
- **Optional hardware output.** An Arduino Leonardo can act as a real USB keyboard and mouse.

---

## Quick start (macOS)

Requirements: a Mac (Apple Silicon recommended), a webcam, Minecraft Education Edition, and either Anaconda/Miniconda or Python 3.10–3.12.

```bash
git clone https://github.com/<your-username>/minecraft-biogaming.git
cd minecraft-biogaming
```

Then double-click the launchers in Finder, in order:

| Launcher | What it does |
|---|---|
| **`1_Setup.command`** | One-time. Creates a native (arm64) Python environment in `.venv` (via conda, or plain `venv` as a fallback), installs `requirements.txt` and downloads the two MediaPipe models (~11 MB). |
| **`2_Output_Test.command`** | No camera needed. Click into a Minecraft world within 8 s and a scripted virtual player walks, looks, jumps, mines and places for ~40 s. Confirms key/mouse output works. |
| **`3_Play.command`** | **The real controller** (face + hand). Calibrates, then starts paused. |
| **`4_Play_FaceOnly.command`** | The hands-free profile. |
| `5_Diagnose_Output.command` | Step-by-step test of every key, button and mouse method (not focus-locked, so keep Minecraft in front). |

If macOS says a launcher can't be opened, right-click it → **Open** → **Open**.

### macOS permissions
In **System Settings → Privacy & Security**, allow **Terminal** (or whichever app runs Python):
- **Camera**, so it can see you;
- **Accessibility**, so it can press keys and move the mouse (without this nothing happens in game);
- **Input Monitoring**, for the global F8 pause hotkey.

Quit and reopen Terminal after changing these.

### Minecraft settings
- Play in **windowed mode** so you can see the HUD beside the game.
- Keep the **default key bindings** (WASD, Space, Shift, Ctrl). If you change them, edit `bindings_keys` in the profile.
- Start with a moderate **mouse sensitivity** and turn **auto-jump off**.
- Practise in a flat **Creative** world first.

---

## Controls

All numbers are **relative to calibration** and are starting points meant to be tuned (see [Tuning](#tuning-the-controls)). "On / off" are the two thresholds of each comparator: the action starts when the signal crosses *on* and only stops once it falls back past *off*.

### Face + hand (default: `profiles/face_hand.yaml`)

**Movement: your dominant palm is a joystick** around its resting spot. 100 % = the comfortable reach you showed during calibration.

| Action | Key | Gesture | On | Off |
|---|---|---|---|---|
| Forward | W | move palm **up** | 35 % of reach | 25 % |
| Back | S | move palm **down** | 35 % | 25 % |
| Strafe right | D | move palm **right** | 35 % | 25 % |
| Strafe left | A | move palm **left** | 35 % | 25 % |

Diagonals work (W + D). If your hand leaves the camera view, all movement stops.

**Look: your head is a second joystick** (rate control).

| Head angle past neutral (yaw / pitch) | Camera speed |
|---|---|
| ≤ 6° (deadzone) | none, so you can glance around the screen |
| 11° | ~120 px/s |
| 21° | ~450 px/s |
| ≥ 36° | ~1,200 px/s (cap: 40 px/frame, so a tracking glitch can't spin you) |

**Clicks and face actions**

| Action | Input | Gesture | On | Off | Debounce |
|---|---|---|---|---|---|
| Attack / mine (held) | left click | **fist** | curl ≥ 60 % while Flexing/Holding | curl < 40 % (or clearly opening) | state machine |
| Use / place (held) | right click | **thumb–index pinch** | gap < 0.30 × palm size | > 0.45 | 80 ms; ignored if fingers > 50 % curled |
| Jump (held) | Space | **open mouth** | jawOpen > 0.35 | < 0.20 | none |
| Sprint | Ctrl | **smile** (either side) | mouthSmile > 0.45 | < 0.25 | 120 ms |
| Sneak | Shift | **raise eyebrows** | browInnerUp > 0.40 | < 0.25 | 150 ms |
| Pause / resume | | **both palms open** | held 1.0 s | | |

Notes:
- *Curl* is the average bend of the index–little fingers: 0 % = flat hand, 100 % = tight fist.
- *Face scores* are MediaPipe blendshapes (0–1) with your resting expression subtracted.
- *Sprint*: Minecraft keeps you sprinting after Ctrl is released for as long as you keep walking forward, so a quick smile is enough.

**The Rest / Flex / Hold / Extend panel** in the HUD labels your hand each frame from curl θ and its rate dθ/dt:

| State | Rule |
|---|---|
| Rest | θ ≤ 25 % and slow |
| Flex | closing faster than 100 %/s |
| Hold | θ > 25 % and roughly still |
| Extend | opening faster than 100 %/s |

The fist (left click) latches on at Flex/Hold with θ ≥ 60 % and releases below 40 %, or early when you're clearly Extending. This rejects partial curls while your hand moves around for walking, and releases quickly when you open your hand.

### Face only (`profiles/face_only.yaml`)
Designed for players without hand use (e.g. bilateral upper-limb difference or spinal cord injury). Because the head can't steer and walk at the same time, it has **two modes**.

| Action | Mode | Gesture | On | Off |
|---|---|---|---|---|
| **Switch WALK ⇄ LOOK** | both | **close both eyes** for 0.6 s (normal blinks are too short) | blink > 0.5 | < 0.3 |
| Strafe right / left | walk | tilt head toward right / left shoulder | roll > 10° | < 6° |
| Forward / back | walk | lean toward / away from camera | face 8 % bigger / smaller | 5 % |
| Look | look | turn / tilt head (same joystick as above) | 6° deadzone | |
| Jump | both | open mouth | jawOpen > 0.35 | < 0.20 |
| Attack / mine | both | raise eyebrows | > 0.40 (120 ms) | < 0.25 |
| Use / place | both | pucker lips **or** puff cheeks | > 0.45 / > 0.40 (120 ms) | < 0.30 / < 0.25 |
| Pause / resume | both | big smile held 1.5 s | > 0.5 | < 0.3 |

Starts in WALK mode. Sneak and sprint are not mapped in this profile.

### Hand only (`profiles/hand_only.yaml`)
| Action | Gesture |
|---|---|
| Switch WALK ⇄ LOOK | pinch with your **other** hand for 0.5 s |
| Move (walk) / look (look) | dominant palm as a joystick |
| Attack / use | dominant fist / pinch |
| Jump | curl the fingers of your other hand |
| Sprint | push your palm almost to the top edge |
| Pause | both palms open for 1 s |

### Keyboard keys (for the operator)
| Key | Effect |
|---|---|
| **F8** (from any app) | Pause / resume (needs Input Monitoring) |
| **P** or **Space** in the camera window | Pause / resume |
| **C** in the camera window | Recalibrate |
| **Esc** / **Q** in the camera window, or **Ctrl-C** in Terminal | Quit (everything is released) |

---

## Calibration
`3_Play.command` and `4_Play_FaceOnly.command` calibrate at every start, because your seating position changes between sessions.

1. **Neutral (2 s).** Look at the screen with a relaxed face, and rest your control hand **beside your face, not in front of it**, where it will sit while you play. The program takes the median of your head angles, nose position, face size, palm position and every blendshape's resting level.
2. **Reach (2.5 s).** Move your hand to a comfortable edge (up, down or sideways) and hold. The 90th-percentile distance from the resting spot becomes "100 % of reach".

If the hand profile can't see your hand during step 1, it says so and retries (up to 3 times) instead of guessing. The circle in the camera window shows your joystick centre. Press **C** at any time to recalibrate. The result is saved to `~/.mcbio/calibration.json` (or `calibration_<subject>.json`).

---

## Safety
- **Minecraft-only output.** Every 0.2 s the controller checks which app is frontmost (via macOS Quartz). If it isn't Minecraft, it releases all keys and buttons and sends nothing until Minecraft is back in front. Terminal prints `[output ON]` / `[output OFF]` on each change. Override with `--any-app` (not recommended).
- **Starts paused**, with a 3-second "STARTING IN" countdown on every resume.
- **Edge-triggered output.** Keys are pressed on rising edges and released on falling edges, and `release_all()` runs on every exit path (Esc, Ctrl-C, exceptions, pause, app switch).
- **Clamped camera speed** (40 px/frame max).
- **Leonardo watchdog.** The Arduino releases everything if it hears nothing for 500 ms.

---

## How it works

One loop runs ~30 times per second:

```
webcam frame
  → tracking.py    MediaPipe HandLandmarker + FaceLandmarker
  → features.py    landmarks → finger curl, pinch, palm position, head yaw/pitch/roll, blendshapes
  → filters.py     One-Euro smoothing
  → calibration    subtract your neutral, divide by your reach        (flatten → dict of signals)
  → gestures.py    fist state machine, pinch, open palm, long blink, smile
  → mapping.py     profile rules → hysteresis comparators → Commands (keys, mouse dx/dy, buttons)
  → outputs.py     press/release changed keys, relative mouse moves  (only if Minecraft is in front)
  → hud.py / logger.py
```

1. **Tracking.** The camera frame is mirrored (so moving right looks right on screen) and passed to MediaPipe's **Tasks API**, which returns 21 landmarks per hand plus handedness, and 478 face landmarks plus 52 *blendshape* scores and a 4×4 facial transformation matrix. Results go into a `Features` dataclass (`types.py`). Nothing after this stage touches the camera, so the whole pipeline can be tested with fake data (`synthetic.py`).
2. **Features.**
   - *Finger curl*: at each finger joint the bend angle between the incoming and outgoing bone vectors (3-D, using MediaPipe's depth estimate) is divided by that joint's maximum flexion and averaged. This is invariant to hand size, distance and rotation.
   - *Palm position*: the centre of the wrist and four knuckles, which barely moves as the fingers bend.
   - *Pinch*: thumb-tip to index-tip distance divided by palm size, so it doesn't depend on distance from the camera.
   - *Head pose*: yaw = atan2(R₀₂, R₂₂), pitch = asin(R₁₂), roll = atan2(−R₁₀, R₁₁) from the rotation matrix.
3. **Filtering.** The One-Euro filter is a first-order low-pass whose cutoff rises with signal speed: heavy smoothing when you're still (no jitter), light smoothing when you move fast (little lag).
4. **Conditioning.** `features.flatten()` subtracts the calibrated neutral and normalises palm offsets by your reach, producing a flat dictionary of signed signals such as `hand.palm_dy`, `face.yaw` and `face.blend.jawOpen`.
5. **Decision.** Each profile rule (`{action, source, op, on, off, min_hold_ms}`) feeds its source signal through a hysteresis detector. Several rules on one action are OR-ed (e.g. either side of a smile). Look is rate control: speed = gain × e × ((1 − expo) + expo × e / e_ref), where e is the angle beyond the deadzone, capped per frame.
6. **Output.** The output backend compares this frame's commands with the last and sends only the changes. Mouse motion is sent as *relative* Quartz events (Minecraft locks the cursor and reads deltas), with sub-pixel remainders carried forward for smooth slow turns.

---

## Project structure
```
minecraft_biogaming/
├── 1_Setup.command … 5_Diagnose_Output.command   double-click launchers (macOS)
├── _env.sh                     shared launcher environment
├── requirements.txt
├── SPEC.md                     design spec / module contract
├── mcbio/
│   ├── types.py                shared dataclasses (Features, Commands, Calibration)
│   ├── tracking.py             camera + MediaPipe hand/face trackers
│   ├── features.py             landmarks → curl, pinch, palm, head pose; flatten()
│   ├── filters.py              One-Euro filter, EMA
│   ├── calibration.py          neutral + reach capture, save/load
│   ├── gestures.py             hysteresis, Rest/Flex/Hold/Extend curl state machine, hold timers
│   ├── mapping.py              YAML profile → Commands (bindings, look, modes, pause)
│   ├── outputs.py              pynput/Quartz, Arduino Leonardo, dry-run, focus gate
│   ├── hud.py                  OpenCV camera window overlay
│   ├── logger.py               per-frame + event CSVs, cue protocol
│   ├── synthetic.py            scripted virtual player for tests/demos
│   └── app.py                  main loop, states, CLI
├── profiles/                   face_hand.yaml, face_only.yaml, hand_only.yaml
├── arduino/leonardo_hid/       Leonardo USB-HID sketch
├── scripts/                    download_models.py, analyze_session.py, diagnose_output.py
├── models/                     MediaPipe .task files (downloaded, not in git)
└── tests/                      pytest suite (synthetic data, no camera)
```

### Libraries
| Library | Used for |
|---|---|
| [MediaPipe](https://developers.google.com/mediapipe) (Tasks API, ≥ 0.10.14) | Hand and face landmark tracking, blendshapes, head pose |
| OpenCV (`opencv-python`) | Camera capture and the HUD window |
| NumPy | Geometry (joint angles, rotation matrices) |
| pynput | Key/mouse output and the F8 hotkey |
| pyobjc-framework-Quartz | Relative mouse events and frontmost-app detection on macOS |
| PyYAML | Control profiles |
| pyserial | Optional Arduino Leonardo backend |
| matplotlib | Analysis plots |
| pytest | Tests |

---

## Command-line usage
Activate the environment (`source .venv/bin/activate`) or use `.venv/bin/python`:

```bash
python -m mcbio                                    # face_hand profile, pynput output
python -m mcbio --profile profiles/face_only.yaml --calibrate
python -m mcbio --backend leonardo --port auto     # Arduino Leonardo as USB keyboard/mouse
python -m mcbio --dry-run                          # everything except sending input
python -m mcbio --synthetic --dry-run --no-hud --frames 300   # no camera, no models
```

| Flag | Meaning |
|---|---|
| `--profile PATH` | Control profile (default `profiles/face_hand.yaml`) |
| `--backend pynput\|leonardo\|dry`, `--dry-run` | Output device |
| `--port PORT\|auto` | Serial port for the Leonardo |
| `--camera N`, `--size WxH` | Camera index and resolution |
| `--calibrate`, `--calib PATH`, `--hand Left\|Right` | Calibration options |
| `--target-app NAME`, `--any-app` | Focus lock (default: only while an app containing "minecraft" is in front) |
| `--no-hud`, `--no-hotkey`, `--quiet` | Headless / no F8 listener / summary only |
| `--synthetic [--seed N] [--realtime]` | Scripted virtual player (no camera or MediaPipe) |
| `--frames N`, `--start-delay S` | Stop after N frames / arming countdown |
| `--log-dir DIR`, `--subject ID` | Session logging |
| `--protocol [--trials N]` | Reaction-time experiment |

---

## Tuning the controls
Edit the profile YAML; no Python changes are needed. Each binding looks like:

```yaml
- {action: forward, source: hand.palm_dy, op: ">", on: 0.35, off: 0.25}
- {action: sprint,  source: face.blend.mouthSmileLeft, op: ">", on: 0.45, off: 0.25, min_hold_ms: 120}
```

- `op ">"`: ON when value > `on`, OFF when value < `off` (so `off ≤ on`). `op "<"` is the mirror image.
- The gap between `on` and `off` is the hysteresis band; widen it if a control flickers.
- `min_hold_ms` / `min_release_ms` add debounce time.
- Two bindings with the same `action` are OR-ed.
- `mode: walk|look` limits a binding to one mode (with a `mode_toggle` section).
- `look:` sets the deadzone, gain, expo and cap for the camera.
- `gestures:` overrides fist, pinch and open-palm thresholds.

Available sources include `face.yaw|pitch|roll`, `face.blend.<blendshape>`, `face.nose_dx|nose_dy`, `face.scale_ratio`, `hand.present`, `hand.palm_dx|palm_dy`, `hand.curl_mean`, `hand.curl_<finger>`, `hand.pinch`, `offhand.*`, `hands.count` and the gestures `gesture.fist|pinch|both_open|both_blink|smile`.

Use the HUD bars to tune: if an action won't trigger, check whether its bar actually reaches the threshold tick.

---

## Arduino Leonardo option (hardware HID)
A Leonardo, Micro or Pro Micro (ATmega32U4) enumerates as a real USB keyboard and mouse, which avoids macOS Accessibility permissions and makes the output stage physical hardware.

1. Open `arduino/leonardo_hid/leonardo_hid.ino` in the Arduino IDE, choose **Arduino Leonardo** and upload.
2. Run `python -m mcbio --backend leonardo --port auto`.
3. Protocol (115200 baud, one line per change): `K+w` / `K-w` key down/up, `M dx dy` relative move (split into ±127 steps), `B+L` / `B-L` / `B+R` / `B-R` buttons, `R` release all, `P` → `OK` ping. A 500 ms watchdog releases everything if the computer stops talking.

---

## Experiment protocol (5 subjects)
For the graduate requirement (accuracy and speed, mean ± SD):

1. Calibrate each participant: `python -m mcbio --calibrate --subject S1`.
2. Let them practise for ~2 minutes in a Creative world.
3. Run the cue protocol:
   ```bash
   python -m mcbio --protocol --subject S1 --log-dir sessions/ --trials 33
   ```
   Big cues appear in the HUD (FORWARD, BACK, LEFT, RIGHT, JUMP, ATTACK, INTERACT, LOOK LEFT, LOOK RIGHT, REST) for a random 2–4 s each (REST 3–6 s), in shuffled balanced blocks.
   - **Reaction time**: from cue onset to the first new action onset.
   - **Accuracy**: whether that first onset matches the cue (otherwise wrong or miss).
   - **False activations**: any onsets during REST, per minute of rest.
4. Repeat for S2–S5, then run `python scripts/analyze_session.py sessions/`.

The analysis prints and saves per-subject accuracy, median and mean ± SD reaction time, false activations per minute, actions per minute, latency p50/p95 and fps, plus per-action mean ± SD across subjects and plots (`rt_by_action.png`, `latency_hist.png`).

Each session folder contains `session.json`, `frames.csv`, `events.csv` and `cues.csv`. Session data is git-ignored so participant recordings stay local.

Try the analysis without people (the synthetic player has human-like delays and occasional mistakes):
```bash
for i in 1 2 3 4 5; do python -m mcbio --synthetic --dry-run --no-hud --protocol --subject S$i --seed $i --log-dir demo_sessions --quiet; done
python scripts/analyze_session.py demo_sessions
```

---

## Testing
```bash
python -m pytest -q
```
123 tests run on synthetic data, with no camera, models or real key presses. They cover:
- curl and head-pose math;
- One-Euro filtering;
- hysteresis and debounce;
- the curl state machine, and that a fist isn't counted as a pinch;
- profile parsing;
- edge-triggered output and guaranteed release;
- the focus lock (nothing is sent while another app is in front);
- calibration following your real hand position;
- logging and the full app loop.

---

## Troubleshooting
| Symptom | Fix |
|---|---|
| Wrong camera / "cannot open camera" | Try `--camera 1` (an iPhone Continuity Camera can take index 0). Close Zoom, FaceTime or Photo Booth. Check the Camera permission. |
| HUD indicators light up but nothing happens in game | Grant **Accessibility** to Terminal and restart it. Make sure Minecraft is in front (Terminal shows `[output ON]`). |
| It typed into another app | It shouldn't: input is focus-locked to Minecraft. Check you didn't pass `--any-app`. |
| F8 does nothing | Grant **Input Monitoring**, or press P in the camera window. |
| Camera drifts or spins | Recalibrate (C) looking straight at the screen. Raise `look.deadzone_deg` or lower `gain_px_per_deg_s`. |
| Walking without meaning to | Recalibrate with your hand where it really rests. Raise the palm `on`/`off` thresholds. |
| Hand covers your face | Rest your hand beside your face during calibration; the joystick centre follows it. |
| Jitter, or tracking lost | Light your face from the front, use a plain background, and sit 50–80 cm from the camera. |
| `models/... not found` | Run `python scripts/download_models.py`. |
| Setup: `mediapipe` version not found | Your Python is x86 (Rosetta). `1_Setup.command` builds a native arm64 env automatically; delete `.venv` and rerun it. |
| Low fps | Use `--size 960x540`, close other camera apps and plug in the laptop. |

---

## Mapping to course concepts
| Instrumentation stage | In mcbio | Course concept |
|---|---|---|
| Sensor | Webcam CMOS at 30 fps | Transduction, sampling rate, Nyquist limit for gesture speed |
| Feature extraction | MediaPipe landmarks → curl, pinch, palm, head pose, blendshapes | Turning a raw signal into a physiological variable; normalising by palm/face size, like a reference electrode |
| Signal conditioning | Subtract the calibrated neutral, divide by reach | Offset nulling and gain (zero and span of an instrumentation amp) |
| Filtering | One-Euro adaptive low-pass | Noise bandwidth vs. phase-lag trade-off |
| Decision | Hysteresis comparators, debounce, curl state machine | Schmitt trigger, switch debouncing, EMG-style onset detection |
| Rate control | Deadzone + proportional speed + saturation | Deadband, P-control, safety clamp |
| Output | Edge-triggered HID events (software or Leonardo) | Output stage, fail-safe design (watchdog, armed start, release on fault) |
| Verification | Latency, fps, reaction time, accuracy, false-activation rate | Instrument characterisation: sensitivity and specificity |

---

## Acknowledgements
- Hand and face tracking: Google **MediaPipe** Tasks (HandLandmarker, FaceLandmarker).
- One-Euro filter: Casiez, Roussel & Vogel, *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems*, CHI 2012.
- The Rest/Flex/Hold/Extend curl state machine is adapted from the authors' earlier BCI experiment code (`dual/src/states.py`).
- Built for EN.580.471/771 Biomedical Instrumentation, Johns Hopkins University.
