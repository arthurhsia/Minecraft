"""Global F8 hotkey: macOS uses a Quartz event tap (no pynput Listener, which crashed newer macOS)."""
import threading
import time

from mcbio.app import MAC_KEYCODES, GlobalHotkey


class FakeQuartz:
    """Just enough of pyobjc's Quartz for the event tap: records calls, runs a fake run loop."""
    kCGSessionEventTap, kCGHeadInsertEventTap, kCGEventTapOptionListenOnly = 1, 0, 1
    kCGEventKeyDown, kCGKeyboardEventKeycode, kCGKeyboardEventAutorepeat = 10, 9, 8
    kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput = 0xFFFFFFFE, 0xFFFFFFFF
    kCFRunLoopCommonModes = "common"

    def __init__(self, grant=True):
        self.grant = grant
        self.callback = None
        self.enabled = []
        self.stopped = threading.Event()

    def CGEventMaskBit(self, t):
        return 1 << t

    def CGEventTapCreate(self, tap, place, options, mask, callback, refcon):
        assert options == self.kCGEventTapOptionListenOnly and mask == 1 << self.kCGEventKeyDown
        self.callback = callback
        return "tap" if self.grant else None

    def CFMachPortCreateRunLoopSource(self, alloc, port, order):
        return "src"

    def CFRunLoopGetCurrent(self):
        return "loop"

    def CFRunLoopAddSource(self, loop, src, mode):
        pass

    def CGEventTapEnable(self, tap, on):
        self.enabled.append(on)

    def CFRunLoopRun(self):
        self.stopped.wait(5)

    def CFRunLoopStop(self, loop):
        self.stopped.set()

    def CGEventGetIntegerValueField(self, event, field):
        return event[field]


def key(code, repeat=0):
    return {FakeQuartz.kCGKeyboardEventKeycode: code, FakeQuartz.kCGKeyboardEventAutorepeat: repeat}


def test_mac_hotkey_uses_event_tap_and_detects_f8():
    q = FakeQuartz()
    hk = GlobalHotkey("f8", platform="darwin", quartz=q)
    assert hk.start() and hk.ok
    cb = q.callback
    cb(None, q.kCGEventKeyDown, key(0), None)              # 'a': ignored
    assert not hk.poll()
    cb(None, q.kCGEventKeyDown, key(MAC_KEYCODES["f8"], repeat=1), None)   # auto-repeat: ignored
    assert not hk.poll()
    ev = key(MAC_KEYCODES["f8"])
    assert cb(None, q.kCGEventKeyDown, ev, None) is ev     # listen-only: event passed through
    assert hk.poll() and not hk.poll()                     # once per press
    cb(None, q.kCGEventTapDisabledByTimeout, None, None)   # macOS disabled the tap -> re-enable
    assert q.enabled[-1] is True
    hk.stop()
    assert q.stopped.wait(1) and q.enabled[-1] is False


def test_mac_hotkey_without_input_monitoring_degrades():
    hk = GlobalHotkey("f8", platform="darwin", quartz=FakeQuartz(grant=False))
    assert hk.start() is False and not hk.ok
    assert "Input Monitoring" in hk.error
    hk.stop()                                              # still safe


def test_mac_hotkey_never_imports_pynput_listener(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def guard(name, *a, **k):
        assert not name.startswith("pynput"), "pynput Listener must not be used on macOS"
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    hk = GlobalHotkey("f8", platform="darwin", quartz=FakeQuartz())
    assert hk.start()
    hk.stop()
