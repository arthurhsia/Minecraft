"""macOS privacy-permission checks for the start screen (ctypes only, no extra packages).

The controller needs three permissions for whichever app runs it (the packaged
"Minecraft Biogaming" app, or Terminal when run from source):

* **Camera** - asked for automatically the first time the camera is opened (the app's
  Info.plist carries the reason text). There is no cheap check here; the start screen
  simply retries opening the camera until it delivers frames.
* **Accessibility** - needed to press keys and move the mouse. ``accessibility()``
  checks it; ``prompt_accessibility()`` adds the app to the list and shows the system
  prompt.
* **Input Monitoring** - needed for the global F8 pause hotkey. ``input_monitoring()``
  checks it; ``prompt_input_monitoring()`` asks for it.

Every function returns ``None`` (unknown) instead of raising when it cannot tell,
e.g. on another OS or if a framework call is unavailable.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys

_AS = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
_CF = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_IOKIT = "/System/Library/Frameworks/IOKit.framework/IOKit"
_HID_LISTEN_EVENT = 1          # kIOHIDRequestTypeListenEvent

SETTINGS_URLS = {
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "input_monitoring": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
}


def _lib(path: str):
    if sys.platform != "darwin":
        return None
    try:
        return ctypes.cdll.LoadLibrary(path)
    except OSError:
        return None


def accessibility() -> bool | None:
    """True if this process may post key/mouse events (Accessibility granted)."""
    lib = _lib(_AS)
    if lib is None:
        return None
    try:
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(lib.AXIsProcessTrusted())
    except (AttributeError, OSError):
        return None


def prompt_accessibility() -> bool | None:
    """Show the system "allow this app to control your computer" prompt; returns current state."""
    ax, cf = _lib(_AS), _lib(_CF)
    if ax is None or cf is None:
        return None
    try:
        key = ctypes.c_void_p.in_dll(ax, "kAXTrustedCheckOptionPrompt")
        true = ctypes.c_void_p.in_dll(cf, "kCFBooleanTrue")
        key_cb = ctypes.c_char.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")
        val_cb = ctypes.c_char.in_dll(cf, "kCFTypeDictionaryValueCallBacks")
        keys = (ctypes.c_void_p * 1)(key.value)
        vals = (ctypes.c_void_p * 1)(true.value)
        cf.CFDictionaryCreate.restype = ctypes.c_void_p
        cf.CFDictionaryCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
        opts = cf.CFDictionaryCreate(None, keys, vals, 1, ctypes.addressof(key_cb),
                                     ctypes.addressof(val_cb))
        ax.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
        ax.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]
        granted = bool(ax.AXIsProcessTrustedWithOptions(opts))
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        cf.CFRelease(opts)
        return granted
    except (AttributeError, OSError, ValueError):
        return accessibility()


def input_monitoring() -> bool | None:
    """True if granted, False if denied or not yet asked, None if unknown."""
    lib = _lib(_IOKIT)
    if lib is None:
        return None
    try:
        lib.IOHIDCheckAccess.restype = ctypes.c_int
        lib.IOHIDCheckAccess.argtypes = [ctypes.c_int]
        return lib.IOHIDCheckAccess(_HID_LISTEN_EVENT) == 0     # 0 = granted, 1 = denied, 2 = unknown
    except (AttributeError, OSError):
        return None


def prompt_input_monitoring() -> bool | None:
    """Ask for Input Monitoring (shows the system prompt the first time)."""
    lib = _lib(_IOKIT)
    if lib is None:
        return None
    try:
        lib.IOHIDRequestAccess.restype = ctypes.c_bool
        lib.IOHIDRequestAccess.argtypes = [ctypes.c_int]
        return bool(lib.IOHIDRequestAccess(_HID_LISTEN_EVENT))
    except (AttributeError, OSError):
        return None


def open_settings(which: str) -> bool:
    """Open System Settings at the given privacy pane ("camera", "accessibility", ...)."""
    url = SETTINGS_URLS.get(which)
    if url is None or sys.platform != "darwin":
        return False
    try:
        subprocess.Popen(["open", url])
        return True
    except OSError:
        return False
