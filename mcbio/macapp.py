"""Small macOS app integrations for the packaged app: a Quit menu and a pynput safety net.

* :func:`install_quit_menu` - OpenCV's window does not create a menu bar, so the app
  had no "Quit" item and Cmd-Q did nothing. This adds an app menu with
  "Quit Minecraft Biogaming  Cmd-Q". Choosing it only sets a flag
  (:func:`quit_requested`); the main loop then stops normally, so every key and mouse
  button is released before the app exits.
* :func:`harden_pynput` - pynput looks up the keyboard layout
  (``TSMGetInputSourceProperty``) whenever its keyboard listener starts, on the
  listener's background thread. Recent macOS kills the process for that. We no longer
  use pynput's listener on macOS, but as a safety net this reads the layout once on the
  main thread and makes pynput reuse it, so no code path can trigger that crash.

Everything is a no-op returning False outside macOS or if pyobjc / pynput is missing.
"""
from __future__ import annotations

import contextlib
import sys
import threading
from typing import Any

from mcbio import paths

_quit = threading.Event()
_keep: list[Any] = []            # Objective-C objects that must stay alive (menu, target)
_target_cls: Any = None


def quit_requested() -> bool:
    """True once the user chose Quit (menu or Cmd-Q)."""
    return _quit.is_set()


def request_quit() -> None:
    _quit.set()


def reset_quit() -> None:
    _quit.clear()


def _quit_target_class() -> Any:
    """Objective-C class whose ``quit:`` action sets the quit flag (defined once)."""
    global _target_cls
    if _target_cls is None:
        from Foundation import NSObject

        class MCBioQuitTarget(NSObject):          # noqa: N801 - Objective-C naming
            def quit_(self, sender: Any) -> None:  # selector "quit:"
                request_quit()

        _target_cls = MCBioQuitTarget
    return _target_cls


def install_quit_menu() -> bool:
    """Give the app a menu bar with Quit (Cmd-Q). Call on the main thread after the
    OpenCV window exists (OpenCV creates the NSApplication)."""
    if sys.platform != "darwin":
        return False
    try:
        from AppKit import NSApplication, NSMenu, NSMenuItem

        app = NSApplication.sharedApplication()
        target = _quit_target_class().alloc().init()
        main_menu = NSMenu.alloc().init()
        app_item = NSMenuItem.alloc().init()
        main_menu.addItem_(app_item)
        app_menu = NSMenu.alloc().initWithTitle_(paths.APP_NAME)
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"Quit {paths.APP_NAME}", "quit:", "q")
        quit_item.setTarget_(target)
        app_menu.addItem_(quit_item)
        app_item.setSubmenu_(app_menu)
        app.setMainMenu_(main_menu)
        _keep[:] = [target, main_menu, app_item, app_menu, quit_item]
        return True
    except Exception as e:   # noqa: BLE001 - a missing menu must never stop the app
        print(f"[menu] could not add the Quit menu: {e}", flush=True)
        return False


def harden_pynput() -> bool:
    """Read the keyboard layout once, here on the main thread, and make pynput reuse it."""
    if sys.platform != "darwin":
        return False
    try:
        import pynput._util.darwin as util
        import pynput.keyboard._darwin as kb

        with util.keycode_context() as ctx:     # (keyboard type, layout bytes): plain values
            cached = ctx

        @contextlib.contextmanager
        def cached_keycode_context():           # same interface, no system call
            yield cached

        util.keycode_context = cached_keycode_context
        kb.keycode_context = cached_keycode_context
        return True
    except Exception as e:   # noqa: BLE001
        print(f"[pynput] keyboard-layout cache not installed: {e}", flush=True)
        return False
