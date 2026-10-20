"""Binary gesture detectors built on continuous features.

Everything here is a small, stateful, *time-aware* object that is fed one value per
frame together with the frame timestamp ``t`` (monotonic seconds).  Using real
timestamps (instead of counting frames) keeps behaviour identical whether the camera
runs at 15 fps or 60 fps.

* :class:`Hysteresis` – Schmitt trigger with optional debounce times.  This is what
  stops keys from "chattering" when a noisy signal hovers near a threshold.
* :class:`CurlStateMachine` – labels the hand as Rest / Flex / Hold / Extend from the
  finger-curl magnitude θ and its velocity θ̇ (the same Eq. 3 labelling as the
  natural-movement prior in ``dual/src/states.py``), and latches a "fist closed"
  output used for attack / mine.
* :class:`HoldTimer` – fires once when a condition has been true continuously for
  N seconds (pause gesture, look/walk mode toggle).
"""
from __future__ import annotations

import math

STATES: tuple[str, ...] = ("Rest", "Flex", "Hold", "Extend")


class Hysteresis:
    """Two-threshold binary detector with optional minimum hold / release times.

    For ``op=">"`` the output turns ON once ``value > on`` and turns OFF once
    ``value < off`` (so ``off <= on``).  For ``op="<"`` it is mirrored: ON once
    ``value < on``, OFF once ``value > off`` (so ``off >= on``).  Between the two
    thresholds the previous output is kept, which is what removes chatter.

    ``min_hold_ms`` – the ON condition must persist this long before the output turns
    on (debounces brief spikes, e.g. a blink or a tracking glitch).
    ``min_release_ms`` – the OFF condition must persist this long before the output
    turns off (bridges brief dropouts while a gesture is held).
    """

    def __init__(
        self,
        on: float,
        off: float,
        op: str = ">",
        min_hold_ms: float = 0.0,
        min_release_ms: float = 0.0,
    ) -> None:
        if op not in (">", "<"):
            raise ValueError(f"op must be '>' or '<', got {op!r}")
        if op == ">" and off > on:
            raise ValueError(f"for op '>' need off <= on (got on={on}, off={off})")
        if op == "<" and off < on:
            raise ValueError(f"for op '<' need off >= on (got on={on}, off={off})")
        if min_hold_ms < 0 or min_release_ms < 0:
            raise ValueError("min_hold_ms / min_release_ms must be >= 0")
        self.on = float(on)
        self.off = float(off)
        self.op = op
        self.min_hold_s = min_hold_ms / 1000.0
        self.min_release_s = min_release_ms / 1000.0
        self.state = False
        self._pending_since: float | None = None   # when the opposite condition started

    def _wants_on(self, value: float) -> bool:
        return value > self.on if self.op == ">" else value < self.on

    def _wants_off(self, value: float) -> bool:
        return value < self.off if self.op == ">" else value > self.off

    def update(self, value: float, t: float) -> bool:
        """Feed one sample; return the (debounced) binary state."""
        switching = self._wants_off(value) if self.state else self._wants_on(value)
        if not switching:
            self._pending_since = None
            return self.state
        if self._pending_since is None:
            self._pending_since = t
        needed = self.min_release_s if self.state else self.min_hold_s
        if t - self._pending_since >= needed:
            self.state = not self.state
            self._pending_since = None
        return self.state

    def reset(self) -> None:
        """Force the output OFF and forget any pending transition."""
        self.state = False
        self._pending_since = None


class CurlStateMachine:
    """Rest / Flex / Hold / Extend labelling of finger curl, plus a fist latch.

    Input θ is a curl magnitude in 0..1 (0 = open hand, 1 = tight fist), e.g. the mean
    curl of index..little.  θ is smoothed with a first-order low-pass (time constant
    ``tau_s``), θ̇ is its finite-difference derivative, smoothed again with
    ``tau_vel_s``.  Each frame is then labelled exactly as in Eq. 3 of the
    natural-movement prior (``dual/src/states.py``)::

        Rest   : θ <= eps_theta  (and slow)
        Flex   : θ̇ >  eps_thetadot          (closing)
        Hold   : θ >  eps_theta and |θ̇| <= eps_thetadot
        Extend : θ̇ < -eps_thetadot          (opening)

    The fist output (:attr:`is_closed`) is a latch on top of those labels:

    * closes when the hand is Flexing or Holding with θ >= ``theta_on``
      (i.e. the Flex -> Hold transition of a real grip);
    * opens when θ < ``theta_off`` (Rest), or early when the hand is clearly
      Extending and θ has dropped below ``theta_on``.

    ``theta_off < theta_on`` gives the hysteresis band.
    """

    def __init__(
        self,
        theta_on: float = 0.6,
        theta_off: float = 0.4,
        eps_theta: float = 0.25,
        eps_thetadot: float = 1.0,
        tau_s: float = 0.04,
        tau_vel_s: float = 0.06,
    ) -> None:
        if not 0.0 <= theta_off <= theta_on:
            raise ValueError("need 0 <= theta_off <= theta_on")
        if eps_thetadot <= 0 or tau_s < 0 or tau_vel_s < 0:
            raise ValueError("eps_thetadot must be > 0 and time constants >= 0")
        self.theta_on = theta_on
        self.theta_off = theta_off
        self.eps_theta = eps_theta
        self.eps_thetadot = eps_thetadot
        self.tau_s = tau_s
        self.tau_vel_s = tau_vel_s
        self.reset()

    @staticmethod
    def _alpha(dt: float, tau: float) -> float:
        """EMA weight for a first-order low-pass with time constant ``tau``."""
        return 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)

    def reset(self) -> None:
        """Forget history (call when the hand disappears)."""
        self.state = "Rest"
        self.is_closed = False
        self.theta = 0.0          # smoothed θ
        self.theta_dot = 0.0      # smoothed θ̇ (units of θ per second)
        self._t: float | None = None

    def update(self, theta: float, t: float) -> str:
        """Feed one curl sample at time ``t``; return the state label."""
        if self._t is None:
            self.theta, self.theta_dot = float(theta), 0.0
        else:
            dt = t - self._t
            if dt <= 0:          # duplicate / out-of-order frame: ignore
                return self.state
            prev = self.theta
            self.theta += self._alpha(dt, self.tau_s) * (theta - self.theta)
            raw_vel = (self.theta - prev) / dt
            self.theta_dot += self._alpha(dt, self.tau_vel_s) * (raw_vel - self.theta_dot)
        self._t = t

        if self.theta_dot > self.eps_thetadot:
            self.state = "Flex"
        elif self.theta_dot < -self.eps_thetadot:
            self.state = "Extend"
        elif self.theta > self.eps_theta:
            self.state = "Hold"
        else:
            self.state = "Rest"

        if not self.is_closed:
            if self.state in ("Flex", "Hold") and self.theta >= self.theta_on:
                self.is_closed = True
        elif self.theta < self.theta_off or (
            self.state == "Extend" and self.theta < self.theta_on
        ):
            self.is_closed = False
        return self.state


class HoldTimer:
    """Rising-edge detector for "condition true continuously for >= ``hold_s``".

    :meth:`update` returns True exactly once per hold; the condition must go false
    again before the timer can fire a second time.
    """

    def __init__(self, hold_s: float) -> None:
        if hold_s < 0:
            raise ValueError("hold_s must be >= 0")
        self.hold_s = float(hold_s)
        self._since: float | None = None
        self._fired = False
        self.progress = 0.0       # 0..1, for a HUD progress ring

    def update(self, cond: bool, t: float) -> bool:
        if not cond:
            self.reset()
            return False
        if self._since is None:
            self._since = t
        elapsed = t - self._since
        self.progress = 1.0 if self.hold_s == 0 else min(1.0, elapsed / self.hold_s)
        if not self._fired and elapsed >= self.hold_s:
            self._fired = True
            return True
        return False

    def reset(self) -> None:
        self._since = None
        self._fired = False
        self.progress = 0.0
