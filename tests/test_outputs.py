"""Tests for mcbio.outputs with fake devices (no real key presses)."""
from types import SimpleNamespace

import pytest

from mcbio.outputs import DryRunOutput, PynputOutput, SerialLeonardoOutput
from mcbio.types import Commands


def C(keys=(), dx=0.0, dy=0.0, lb=False, rb=False, taps=()):
    return Commands(keys=set(keys), mouse_dx=dx, mouse_dy=dy, left_button=lb,
                    right_button=rb, taps=list(taps))


def test_dryrun_is_edge_triggered():
    out = DryRunOutput()
    out.apply(C({"forward"}))
    out.apply(C({"forward"}))                 # still held: nothing new sent
    out.apply(C({"forward", "jump"}, lb=True))
    out.apply(C({"jump"}, lb=True))
    out.apply(C())
    assert out.events == [
        ("key", "w", True),
        ("key", "space", True), ("button", "L", True),
        ("key", "w", False),
        ("key", "space", False), ("button", "L", False),
    ]
    out.close()


def test_dryrun_accumulates_fractional_mouse():
    out = DryRunOutput()
    for _ in range(4):
        out.apply(C(dx=0.4, dy=-0.3))
    # x: 0.4, 0.8, 1.2 -> send 1 (keep 0.2), 0.6;  y: ... -1.2 on frame 4 -> send -1.
    assert out.events == [("move", 1, 0), ("move", 0, -1)]
    out.close()


def test_taps_press_and_release():
    out = DryRunOutput()
    out.apply(C(taps=["inventory", "hotbar_next"]))
    assert out.events == [("key", "e", True), ("key", "e", False), ("scroll", -1)]
    out.close()


def test_release_all_after_held_keys_is_idempotent():
    out = DryRunOutput({"sprint": "ctrl"})
    out.apply(C({"forward", "sprint"}, rb=True))
    out.events.clear()
    out.release_all()
    assert sorted(out.events) == sorted([("key", "w", False), ("key", "ctrl", False),
                                         ("button", "R", False)])
    out.events.clear()
    out.release_all()
    out.close()
    out.close()
    assert out.events == []
    out.apply(C({"forward"}))                 # closed: ignored
    assert out.events == []


def test_close_releases_held():
    out = DryRunOutput()
    out.apply(C({"back"}, lb=True))
    out.close()
    assert ("key", "s", False) in out.events and ("button", "L", False) in out.events


def test_unknown_key_name_rejected():
    with pytest.raises(ValueError):
        DryRunOutput({"forward": "wasd"})


class FakeSerial:
    def __init__(self):
        self.written = b""
        self.in_waiting = 0
        self.closed = False

    def write(self, data):
        self.written += data

    def read(self, n):
        return b""

    def close(self):
        self.closed = True

    def lines(self):
        out = self.written.decode().split("\n")
        assert out[-1] == ""                  # every line newline-terminated
        self.written = b""
        return out[:-1]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_serial_protocol_lines():
    ser, clock = FakeSerial(), FakeClock()
    out = SerialLeonardoOutput(serial_obj=ser, clock=clock)
    out.apply(C({"forward", "jump", "sneak", "sprint"}))
    assert sorted(ser.lines()) == sorted(["K+w", "K+space", "K+shift", "K+ctrl"])
    out.apply(C({"forward", "jump", "sneak", "sprint"}))
    assert ser.lines() == []                  # no change, keepalive not due yet
    out.apply(C({"forward"}, lb=True, rb=True))
    assert sorted(ser.lines()) == sorted(["K-space", "K-shift", "K-ctrl", "B+L", "B+R"])
    out.apply(C({"forward"}, rb=True, taps=["inventory"]))
    assert ser.lines() == ["B-L", "K+e", "K-e"]
    out.apply(C(dx=300, dy=-10))
    assert ser.lines() == ["K-w", "B-R", "M 127 -10", "M 127 0", "M 46 0"]
    out.apply(C(dx=-5, dy=200.7))
    assert ser.lines() == ["M -5 127", "M 0 73"]


def test_serial_keepalive_and_release():
    ser, clock = FakeSerial(), FakeClock()
    out = SerialLeonardoOutput(serial_obj=ser, clock=clock)
    out.apply(C({"left"}))
    assert ser.lines() == ["K+a"]
    clock.t = 0.1
    out.apply(C({"left"}))
    assert ser.lines() == []
    clock.t = 0.25
    out.apply(C({"left"}))
    assert ser.lines() == ["P"]
    out.release_all()
    out.release_all()
    assert ser.lines() == ["R", "R"]
    out.close()
    assert ser.lines() == ["R"] and ser.closed
    out.close()
    assert ser.lines() == []


class FakeController:
    def __init__(self):
        self.calls = []

    def press(self, k):
        self.calls.append(("press", k))

    def release(self, k):
        self.calls.append(("release", k))

    def move(self, dx, dy):
        self.calls.append(("move", dx, dy))

    def scroll(self, dx, dy):
        self.calls.append(("scroll", dx, dy))


KEY_NS = SimpleNamespace(space="<space>", shift="<shift>", ctrl="<ctrl>", esc="<esc>")
BUTTON_NS = SimpleNamespace(left="<L>", right="<R>")


def make_pynput():
    kb, ms = FakeController(), FakeController()
    out = PynputOutput(keyboard=kb, mouse=ms, key_ns=KEY_NS, button_ns=BUTTON_NS,
                       use_quartz=False)
    return out, kb, ms


def test_pynput_key_mapping_and_mouse():
    out, kb, ms = make_pynput()
    assert out.pynput_key("w") == "w" and out.pynput_key("space") == "<space>"
    out.apply(C({"forward", "jump", "sneak"}, dx=3.5, dy=-2, rb=True, taps=["hotbar_prev"]))
    assert sorted(kb.calls) == sorted([("press", "w"), ("press", "<space>"), ("press", "<shift>")])
    assert ms.calls == [("press", "<R>"), ("move", 3, -2), ("scroll", 0, 1)]
    kb.calls.clear()
    ms.calls.clear()
    out.close()
    assert sorted(kb.calls) == sorted([("release", "w"), ("release", "<space>"),
                                       ("release", "<shift>")])
    assert ms.calls == [("release", "<R>")]


def test_outputs_import_does_not_load_pynput_or_serial():
    # pynput needs a display at import time, so it must only be imported lazily.
    import subprocess
    import sys
    code = ("import sys, mcbio.outputs; "
            "assert 'pynput' not in sys.modules and 'serial' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


class FakeQuartz:
    """Just enough of pyobjc's Quartz to check which event types are posted."""
    kCGEventMouseMoved, kCGEventLeftMouseDragged, kCGEventRightMouseDragged = "move", "ldrag", "rdrag"
    kCGMouseButtonLeft, kCGMouseButtonRight = 0, 1
    kCGMouseEventDeltaX, kCGMouseEventDeltaY = "dX", "dY"
    kCGHIDEventTap = "hid"

    def __init__(self):
        self.posted = []

    def CGEventCreate(self, _):
        return None

    def CGEventGetLocation(self, _):
        return SimpleNamespace(x=100.0, y=100.0)

    def CGMainDisplayID(self):
        return 1

    def CGDisplayBounds(self, _):
        return SimpleNamespace(origin=SimpleNamespace(x=0, y=0),
                               size=SimpleNamespace(width=1440, height=900))

    def CGEventCreateMouseEvent(self, _, etype, pos, btn):
        return {"type": etype, "pos": pos}

    def CGEventSetIntegerValueField(self, ev, field, v):
        ev[field] = v

    def CGEventPost(self, tap, ev):
        self.posted.append(ev)


def test_pynput_quartz_relative_and_drag_events():
    out, _, _ = make_pynput()
    q = FakeQuartz()
    out._quartz = q
    out.apply(C(dx=5, dy=-3))
    out.apply(C(dx=2000, lb=True))
    assert q.posted[0] == {"type": "move", "pos": (105.0, 97.0), "dX": 5, "dY": -3}
    assert q.posted[1]["type"] == "ldrag" and q.posted[1]["pos"] == (1439, 100.0)
    out.close()
