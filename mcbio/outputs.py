"""Output backends: turn :class:`~mcbio.types.Commands` into key / mouse events.

All backends share the edge-triggered logic in :class:`OutputBackend`:

* a key / button is **pressed** on the frame it appears in ``Commands`` and
  **released** on the frame it disappears (nothing is re-sent while it stays held);
* mouse motion is sent every frame; fractional pixels are accumulated so slow,
  sub-pixel camera speeds still move;
* ``taps`` are a press immediately followed by a release;
* :meth:`~OutputBackend.release_all` lets go of everything and is safe to call any
  number of times; :meth:`~OutputBackend.close` calls it and is registered with
  ``atexit`` so a crash or Ctrl-C never leaves W (or the mouse button) stuck down.

Backends
--------
* :class:`DryRunOutput` – records events in a list (tests, HUD, ``--dry-run``).
* :class:`PynputOutput` – real keyboard/mouse on the computer running the game.
  On macOS the camera needs *relative* mouse deltas, which are posted with Quartz
  CGEvents when pyobjc is installed.  macOS asks for Accessibility (and Input
  Monitoring) permission for the terminal / Python the first time.
* :class:`SerialLeonardoOutput` – sends a tiny text protocol to an Arduino Leonardo
  (``arduino/leonardo_hid``) that acts as a real USB keyboard + mouse.  Useful when
  the OS blocks synthetic input, and the board's watchdog releases everything if
  this program stops talking to it.
"""
from __future__ import annotations

import atexit
import logging
import math
import time
from typing import Any, Callable

from .types import Commands

log = logging.getLogger(__name__)

#: Logical action -> physical key name (Minecraft defaults). Profiles override this.
DEFAULT_KEYMAP: dict[str, str] = {
    "forward": "w", "back": "s", "left": "a", "right": "d",
    "jump": "space", "sneak": "shift", "sprint": "ctrl",
    "inventory": "e", "hotbar_next": "scroll_down", "hotbar_prev": "scroll_up",
}

#: Pseudo-keys that are sent as mouse-wheel steps (+1 = up).
SCROLL_KEYS = {"scroll_up": 1, "scroll_down": -1}

#: Named (multi-character) keys understood by every backend and the Arduino sketch.
NAMED_KEYS = ("space", "shift", "ctrl", "alt", "cmd", "tab", "enter", "esc",
              "backspace", "up", "down", "left", "right",
              *(f"f{i}" for i in range(1, 13)))

BUTTONS = ("L", "R")


class OutputBackend:
    """Edge-triggered base class. Subclasses implement the ``_`` primitives."""

    def __init__(self, bindings_keys: dict[str, str] | None = None) -> None:
        self.keymap = dict(DEFAULT_KEYMAP)
        if bindings_keys:
            self.keymap.update(bindings_keys)
        for action, key in self.keymap.items():
            if not (len(key) == 1 or key in NAMED_KEYS or key in SCROLL_KEYS):
                raise ValueError(f"action {action!r}: unknown key name {key!r}")
        self.held_keys: set[str] = set()       # logical actions currently pressed
        self.held_buttons: set[str] = set()    # subset of BUTTONS
        self._frac_x = 0.0
        self._frac_y = 0.0
        self._closed = False
        atexit.register(self.close)

    # -- primitives for subclasses ------------------------------------------------------
    def _key(self, key: str, down: bool) -> None:
        """Press (``down=True``) or release a physical key name."""
        raise NotImplementedError

    def _button(self, button: str, down: bool) -> None:
        """Press or release mouse button ``"L"`` / ``"R"``."""
        raise NotImplementedError

    def _move(self, dx: int, dy: int) -> None:
        """Relative mouse motion in whole pixels (dy + = down)."""
        raise NotImplementedError

    def _scroll(self, steps: int) -> None:
        """Mouse wheel; default: unsupported, ignored."""
        log.debug("%s: scroll not supported, ignoring %d", type(self).__name__, steps)

    def _release_everything(self) -> None:
        """Release all held keys and buttons (subclasses may do it in one message).

        Called by every :meth:`release_all`, even when nothing is held."""
        for action in sorted(self.held_keys):
            self._safe(self._key, self.keymap[action], False)
        for b in sorted(self.held_buttons):
            self._safe(self._button, b, False)

    def _tick(self) -> None:
        """Called at the end of every :meth:`apply` (e.g. for keepalives)."""

    def _shutdown(self) -> None:
        """Release backend resources (port, controllers)."""

    # -- helpers ------------------------------------------------------------------------
    @staticmethod
    def _safe(fn: Callable[..., Any], *args: Any) -> None:
        """Call ``fn`` but never let one failed release stop the others."""
        try:
            fn(*args)
        except Exception:                     # noqa: BLE001 - must keep releasing
            log.exception("output call %s%r failed", getattr(fn, "__name__", fn), args)

    # -- public API ---------------------------------------------------------------------
    def apply(self, cmds: Commands) -> None:
        """Send the difference between the previous frame and ``cmds``."""
        if self._closed:
            return
        want_keys = {k for k in cmds.keys if k in self.keymap}
        for action in sorted(self.held_keys - want_keys):
            self._key(self.keymap[action], False)
        for action in sorted(want_keys - self.held_keys):
            self._key(self.keymap[action], True)
        self.held_keys = want_keys

        want_buttons = {b for b, on in (("L", cmds.left_button), ("R", cmds.right_button)) if on}
        for b in sorted(self.held_buttons - want_buttons):
            self._button(b, False)
        for b in sorted(want_buttons - self.held_buttons):
            self._button(b, True)
        self.held_buttons = want_buttons

        # Accumulate sub-pixel motion; send only the whole-pixel part.
        self._frac_x += cmds.mouse_dx
        self._frac_y += cmds.mouse_dy
        ix, iy = math.trunc(self._frac_x), math.trunc(self._frac_y)
        if ix or iy:
            self._frac_x -= ix
            self._frac_y -= iy
            self._move(ix, iy)

        for tap in cmds.taps:
            key = self.keymap.get(tap)
            if key is None:
                log.warning("no key bound for tap %r", tap)
            elif key in SCROLL_KEYS:
                self._scroll(SCROLL_KEYS[key])
            else:
                self._key(key, True)
                self._key(key, False)
        self._tick()

    def release_all(self) -> None:
        """Release every held key and button. Idempotent."""
        self._release_everything()
        self.held_keys = set()
        self.held_buttons = set()
        self._frac_x = self._frac_y = 0.0

    def close(self) -> None:
        """Release everything and free resources. Safe to call more than once."""
        if self._closed:
            return
        self.release_all()
        self._closed = True
        self._safe(self._shutdown)
        atexit.unregister(self.close)

    def __enter__(self) -> "OutputBackend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DryRunOutput(OutputBackend):
    """Records events as tuples instead of sending them.

    ``events`` entries: ``("key", name, down)``, ``("button", "L"|"R", down)``,
    ``("move", dx, dy)``, ``("scroll", steps)``.
    """

    def __init__(self, bindings_keys: dict[str, str] | None = None) -> None:
        super().__init__(bindings_keys)
        self.events: list[tuple] = []

    def _key(self, key: str, down: bool) -> None:
        self.events.append(("key", key, down))

    def _button(self, button: str, down: bool) -> None:
        self.events.append(("button", button, down))

    def _move(self, dx: int, dy: int) -> None:
        self.events.append(("move", dx, dy))

    def _scroll(self, steps: int) -> None:
        self.events.append(("scroll", steps))


class PynputOutput(OutputBackend):
    """Synthetic input on this computer via pynput (+ Quartz relative mouse on macOS).

    Controllers and the ``Key`` / ``Button`` namespaces can be injected (tests);
    otherwise pynput is imported here, lazily, so importing this module never needs a
    display.  ``use_quartz=None`` auto-detects pyobjc's ``Quartz``.
    """

    def __init__(
        self,
        bindings_keys: dict[str, str] | None = None,
        keyboard: Any = None,
        mouse: Any = None,
        key_ns: Any = None,
        button_ns: Any = None,
        use_quartz: bool | None = None,
    ) -> None:
        if None in (keyboard, mouse, key_ns, button_ns):
            from pynput import keyboard as pk, mouse as pm   # lazy: needs a display
            keyboard = keyboard or pk.Controller()
            mouse = mouse or pm.Controller()
            key_ns = key_ns or pk.Key
            button_ns = button_ns or pm.Button
        self._kb, self._mouse = keyboard, mouse
        self._keys, self._buttons = key_ns, button_ns
        self._quartz: Any = None
        if use_quartz is not False:
            try:
                import Quartz  # type: ignore[import-not-found]  # pyobjc, macOS only
                self._quartz = Quartz
            except ImportError:
                if use_quartz:
                    raise
        super().__init__(bindings_keys)

    def pynput_key(self, name: str) -> Any:
        """Physical key name -> pynput key (a ``str`` char or a ``Key`` member)."""
        if len(name) == 1:
            return name
        return getattr(self._keys, name)   # NAMED_KEYS match pynput's Key members

    def _key(self, key: str, down: bool) -> None:
        k = self.pynput_key(key)
        (self._kb.press if down else self._kb.release)(k)

    def _button(self, button: str, down: bool) -> None:
        b = self._buttons.left if button == "L" else self._buttons.right
        (self._mouse.press if down else self._mouse.release)(b)

    def _scroll(self, steps: int) -> None:
        self._mouse.scroll(0, steps)

    def _move(self, dx: int, dy: int) -> None:
        if self._quartz is None:
            self._mouse.move(dx, dy)
            return
        Q = self._quartz
        # Minecraft (with the cursor captured) reads the event's delta fields, not the
        # absolute position, so we set both: a position that moves by the same amount
        # (clamped to the main display so it never runs off-screen) and the deltas.
        cur = Q.CGEventGetLocation(Q.CGEventCreate(None))
        bounds = Q.CGDisplayBounds(Q.CGMainDisplayID())
        x = min(max(cur.x + dx, bounds.origin.x), bounds.origin.x + bounds.size.width - 1)
        y = min(max(cur.y + dy, bounds.origin.y), bounds.origin.y + bounds.size.height - 1)
        if "L" in self.held_buttons:
            etype, btn = Q.kCGEventLeftMouseDragged, Q.kCGMouseButtonLeft
        elif "R" in self.held_buttons:
            etype, btn = Q.kCGEventRightMouseDragged, Q.kCGMouseButtonRight
        else:
            etype, btn = Q.kCGEventMouseMoved, Q.kCGMouseButtonLeft
        ev = Q.CGEventCreateMouseEvent(None, etype, (x, y), btn)
        Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventDeltaX, int(dx))
        Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventDeltaY, int(dy))
        Q.CGEventPost(Q.kCGHIDEventTap, ev)


def find_leonardo_port() -> str | None:
    """Return the serial device of the first Arduino-like board, or None.

    Matches Arduino's USB vendor IDs (0x2341 arduino.cc, 0x2A03 arduino.org) or a
    description mentioning Arduino/Leonardo.  On macOS the ``/dev/cu.*`` device is
    preferred over ``/dev/tty.*`` (``cu`` does not wait for carrier detect).
    """
    from serial.tools import list_ports     # lazy: pyserial

    hits = []
    for p in list_ports.comports():
        text = f"{p.description or ''} {p.manufacturer or ''} {p.product or ''}".lower()
        if p.vid in (0x2341, 0x2A03) or "arduino" in text or "leonardo" in text:
            hits.append(p.device)
    hits.sort(key=lambda d: (not d.startswith("/dev/cu."), d))
    return hits[0] if hits else None


class SerialLeonardoOutput(OutputBackend):
    """Drive an Arduino Leonardo HID bridge over USB serial.

    Protocol (one ASCII command per line, ``\\n``-terminated)::

        K+<key> / K-<key>   key down / up; <key> = single char or a NAMED_KEYS name
        M <dx> <dy>         relative mouse move, each in -127..127 (bigger moves split)
        B+L B-L B+R B-R     mouse button down / up
        R                   release everything
        P                   ping -> board answers "OK"

    Only changes are sent.  If nothing was written for ``keepalive_s`` a ``P`` is sent
    so the board's 500 ms watchdog knows we are alive.  ``serial_obj`` lets tests inject
    a fake port; otherwise pyserial is imported lazily and the port opened (auto-found
    with :func:`find_leonardo_port` when ``port`` is None).
    """

    MAX_STEP = 127

    def __init__(
        self,
        port: str | None = None,
        baud: int = 115200,
        bindings_keys: dict[str, str] | None = None,
        serial_obj: Any = None,
        clock: Callable[[], float] = time.monotonic,
        keepalive_s: float = 0.2,
        handshake_timeout_s: float = 2.0,
    ) -> None:
        self._clock = clock
        self.keepalive_s = keepalive_s
        if serial_obj is None:
            import serial                    # lazy: pyserial
            port = port or find_leonardo_port()
            if port is None:
                raise RuntimeError("no Arduino Leonardo found; pass the port explicitly")
            self._ser = serial.Serial(port, baud, timeout=0.05, write_timeout=0.1)
            if not self.ping(handshake_timeout_s):
                self._ser.close()
                raise RuntimeError(f"{port}: no 'OK' reply - is leonardo_hid.ino flashed?")
        else:
            self._ser = serial_obj
        self._last_write = clock()
        super().__init__(bindings_keys)

    def _send(self, line: str) -> None:
        self._ser.write((line + "\n").encode("ascii"))
        self._last_write = self._clock()

    def ping(self, timeout_s: float = 1.0) -> bool:
        """Send ``P`` and wait up to ``timeout_s`` for an ``OK`` line."""
        self._ser.reset_input_buffer()
        self._send("P")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ser.readline().strip() == b"OK":
                return True
        return False

    def _key(self, key: str, down: bool) -> None:
        self._send(f"K{'+' if down else '-'}{key}")

    def _button(self, button: str, down: bool) -> None:
        self._send(f"B{'+' if down else '-'}{button}")

    def _move(self, dx: int, dy: int) -> None:
        # HID mouse reports are signed bytes, so split large moves into chunks.
        while dx or dy:
            cx = max(-self.MAX_STEP, min(self.MAX_STEP, dx))
            cy = max(-self.MAX_STEP, min(self.MAX_STEP, dy))
            self._send(f"M {cx} {cy}")
            dx -= cx
            dy -= cy

    def _release_everything(self) -> None:
        # One "R" releases every key and button on the board (sent even if we think
        # nothing is held: cheap, and it clears any state left from a previous run).
        if not self._closed:
            self._safe(self._send, "R")

    def _tick(self) -> None:
        if self._clock() - self._last_write >= self.keepalive_s:
            self._send("P")
            # Drain the "OK" replies so the OS input buffer never fills up.
            if getattr(self._ser, "in_waiting", 0):
                self._ser.read(self._ser.in_waiting)

    def _shutdown(self) -> None:
        self._ser.close()


# ---------------------------------------------------------------------------------------
# Focus gating: only drive the game window
# ---------------------------------------------------------------------------------------
def frontmost_app_name() -> str | None:
    """Name of the application owning the frontmost normal window (macOS), else None.

    Uses Quartz's window list (front-to-back order), which is accurate without a Cocoa
    run loop -- unlike NSWorkspace.frontmostApplication, which can go stale in a
    plain Python process.
    """
    try:
        import Quartz as Q  # type: ignore[import-not-found]
    except ImportError:
        return None
    opts = Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements
    for w in Q.CGWindowListCopyWindowInfo(opts, Q.kCGNullWindowID) or []:
        if w.get("kCGWindowLayer", 1) == 0:          # layer 0 = ordinary app windows
            return str(w.get("kCGWindowOwnerName", ""))
    return None


class FocusGatedOutput:
    """Wrap a backend so it only sends input while the game is the frontmost app.

    When another app comes to the front, everything held is released at once and
    nothing more is sent until the game is in front again, so the controller can
    never type into, click on, or move the mouse over other apps.
    """

    def __init__(self, inner: OutputBackend, targets: tuple[str, ...] = ("minecraft",),
                 check_every_s: float = 0.2, probe: Callable[[], str | None] = frontmost_app_name,
                 clock: Callable[[], float] = time.monotonic,
                 on_change: Callable[[bool, str | None], None] | None = None) -> None:
        self.inner = inner
        self.targets = tuple(t.lower() for t in targets)
        self.check_every_s = check_every_s
        self._probe, self._clock, self._on_change = probe, clock, on_change
        self._next_check = 0.0
        self.focused = False
        self.front_app: str | None = None

    def __getattr__(self, name: str) -> Any:          # held_keys, keymap, ... from inner
        return getattr(self.inner, name)

    def _update_focus(self) -> None:
        now = self._clock()
        if now < self._next_check:
            return
        self._next_check = now + self.check_every_s
        name = self._probe()
        self.front_app = name
        focused = name is not None and any(t in name.lower() for t in self.targets)
        if focused != self.focused:
            self.focused = focused
            if not focused:
                self.inner.release_all()
                self.inner._frac_x = self.inner._frac_y = 0.0
            if self._on_change:
                self._on_change(focused, name)

    def apply(self, cmds: Commands) -> None:
        self._update_focus()
        if self.focused:
            self.inner.apply(cmds)

    def release_all(self) -> None:
        self.inner.release_all()

    def close(self) -> None:
        self.inner.close()
