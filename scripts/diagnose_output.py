"""Step-by-step output diagnostic for macOS + Minecraft (no camera needed).

Runs each action ALONE for ~2 s with a numbered banner, so you can see exactly
which ones Minecraft responds to. Mouse-look is tried with four different event
methods, and keys with two methods, because games differ in which synthetic
events they accept. Write down the step numbers that did NOT work.

Usage:  python scripts/diagnose_output.py [--delay 8] [--hold 2.0]
"""
from __future__ import annotations

import argparse
import sys
import time

HOLD = 2.0


def banner(n: int, text: str) -> None:
    print(f"\n=== STEP {n:2d}: {text} ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=8.0)
    ap.add_argument("--hold", type=float, default=HOLD)
    a = ap.parse_args()

    from pynput import keyboard as pk, mouse as pm
    kb, ms = pk.Controller(), pm.Controller()
    try:
        import Quartz as Q
    except ImportError:
        Q = None
        print("NOTE: pyobjc Quartz not available; Quartz steps will be skipped.")

    # macOS virtual key codes for the Quartz keyboard path
    VK = {"w": 13, "a": 0, "s": 1, "d": 2, "space": 49, "shift": 56}

    def q_key(name: str, down: bool) -> None:
        ev = Q.CGEventCreateKeyboardEvent(None, VK[name], down)
        Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def p_key(name: str, down: bool) -> None:
        k = {"space": pk.Key.space, "shift": pk.Key.shift}.get(name, name)
        (kb.press if down else kb.release)(k)

    def hold_key(fn, name: str) -> None:
        fn(name, True); time.sleep(a.hold); fn(name, False)

    def hold_button(btn) -> None:
        ms.press(btn); time.sleep(a.hold); ms.release(btn)

    def look(method: str, dx: int, dy: int) -> None:
        t_end = time.monotonic() + a.hold
        while time.monotonic() < t_end:
            if method == "pynput":
                ms.move(dx, dy)
            else:
                cur = Q.CGEventGetLocation(Q.CGEventCreate(None))
                pos = (cur.x, cur.y) if method == "quartz_still" else (cur.x + dx, cur.y + dy)
                ev = Q.CGEventCreateMouseEvent(None, Q.kCGEventMouseMoved, pos, Q.kCGMouseButtonLeft)
                Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventDeltaX, dx)
                Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventDeltaY, dy)
                tap = Q.kCGSessionEventTap if method == "quartz_session" else Q.kCGHIDEventTap
                Q.CGEventPost(tap, ev)
            time.sleep(1 / 60)

    steps = [
        ("FORWARD (W) via pynput", lambda: hold_key(p_key, "w")),
        ("BACK (S) via pynput", lambda: hold_key(p_key, "s")),
        ("LEFT strafe (A) via pynput", lambda: hold_key(p_key, "a")),
        ("RIGHT strafe (D) via pynput", lambda: hold_key(p_key, "d")),
        ("JUMP (Space) via pynput", lambda: hold_key(p_key, "space")),
        ("SNEAK (Shift) via pynput", lambda: hold_key(p_key, "shift")),
        ("ATTACK / MINE (left mouse button held)", lambda: hold_button(pm.Button.left)),
        ("USE / PLACE (right mouse button held)", lambda: hold_button(pm.Button.right)),
        ("LOOK RIGHT  method A: Quartz HID tap + delta", lambda: look("quartz_hid", 8, 0)),
        ("LOOK LEFT   method B: Quartz session tap + delta", lambda: look("quartz_session", -8, 0)),
        ("LOOK UP     method C: Quartz delta, cursor kept still", lambda: look("quartz_still", 0, -6)),
        ("LOOK DOWN   method D: pynput mouse.move", lambda: look("pynput", 0, 6)),
    ]
    if Q is not None:
        steps += [
            ("FORWARD (W) via Quartz key event", lambda: hold_key(q_key, "w")),
            ("JUMP (Space) via Quartz key event", lambda: hold_key(q_key, "space")),
        ]

    print(f"Click into your Minecraft world NOW. Starting in {a.delay:.0f} s ...", flush=True)
    time.sleep(a.delay)
    try:
        for i, (label, fn) in enumerate(steps, 1):
            banner(i, label)
            if Q is None and "Quartz" in label:
                print("   (skipped: no Quartz)")
                continue
            fn()
            time.sleep(1.2)
    except KeyboardInterrupt:
        pass
    finally:
        for k in ("w", "a", "s", "d", "space", "shift"):
            try:
                p_key(k, False)
            except Exception:
                pass
        for b in (pm.Button.left, pm.Button.right):
            try:
                ms.release(b)
            except Exception:
                pass
    print("\nDone. Tell Claude which STEP numbers did NOT do anything in Minecraft.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
