"""Signal-smoothing filters for noisy landmark-derived features.

Webcam landmark trackers jitter by a few pixels frame-to-frame. A fixed low-pass
filter forces a trade-off: heavy smoothing kills jitter but adds lag during fast
moves, light smoothing is responsive but jittery. The **One Euro filter**
(Casiez, Roussel & Vogel, CHI 2012) resolves this with an *adaptive* cutoff:

    f_c = min_cutoff + beta * |dx/dt|

At rest (|dx/dt| ~ 0) the cutoff is low -> strong smoothing, no jitter.
During fast motion the cutoff rises -> little lag. The derivative itself is
low-pass filtered at a fixed ``d_cutoff`` so noise does not open the filter.

Each stage is a first-order (exponential) low-pass whose smoothing factor for a
sample interval ``dt`` and cutoff ``f_c`` is

    tau   = 1 / (2*pi*f_c)          (RC time constant)
    alpha = 1 / (1 + tau/dt)

which is exact for the discretised RC circuit and handles a variable frame rate.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[float, int, tuple, list, np.ndarray]


def smoothing_factor(dt: float, cutoff_hz: float | np.ndarray) -> float | np.ndarray:
    """First-order low-pass smoothing factor alpha in (0, 1] for interval ``dt``."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter for a scalar or a fixed-length vector.

    Vectors are filtered component-wise (each component gets its own adaptive
    cutoff), as in the reference implementation.

    Parameters
    ----------
    min_cutoff : minimum cutoff frequency in Hz (lower = smoother at rest).
    beta : speed coefficient (higher = less lag on fast motion). Its scale depends
        on the signal units: cutoff grows by ``beta * speed`` where speed is in
        signal-units per second.
    d_cutoff : cutoff (Hz) for the derivative estimate.

    Call ``f(x, t)`` with ``t`` in seconds. Returns a ``float`` for scalar input
    and a ``numpy.ndarray`` for vector input.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be > 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        """Forget all history; the next sample is passed through unchanged."""
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None
        self._scalar = True

    @property
    def value(self) -> float | np.ndarray | None:
        """Last filtered output (None before the first sample)."""
        if self._x is None:
            return None
        return float(self._x) if self._scalar else self._x.copy()

    def __call__(self, x: ArrayLike, t: float) -> float | np.ndarray:
        xa = np.asarray(x, dtype=float)
        if self._x is None or xa.shape != self._x.shape:
            # First sample (or the vector length changed): initialise, no smoothing.
            self._scalar = xa.ndim == 0
            self._x = xa.copy()
            self._dx = np.zeros_like(xa)
            self._t = float(t)
            return self.value  # type: ignore[return-value]

        dt = float(t) - self._t  # type: ignore[operator]
        if dt <= 0.0:
            # Duplicate or out-of-order timestamp: the rate is undefined, so keep
            # the previous estimate rather than dividing by zero.
            return self.value  # type: ignore[return-value]
        self._t = float(t)

        # 1) Low-pass filtered derivative (fixed cutoff).
        dx_raw = (xa - self._x) / dt
        a_d = smoothing_factor(dt, self.d_cutoff)
        self._dx = a_d * dx_raw + (1.0 - a_d) * self._dx

        # 2) Adaptive cutoff from speed, then low-pass the signal itself.
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = smoothing_factor(dt, cutoff)
        self._x = a * xa + (1.0 - a) * self._x
        return self.value  # type: ignore[return-value]


class EMA:
    """Exponential moving average ``y += alpha * (x - y)`` (fixed alpha, per sample).

    ``alpha`` in (0, 1]: 1 = no smoothing. Works for scalars and vectors.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        """Forget history; the next sample is passed through unchanged."""
        self._y: np.ndarray | None = None
        self._scalar = True

    @property
    def value(self) -> float | np.ndarray | None:
        """Last output (None before the first sample)."""
        if self._y is None:
            return None
        return float(self._y) if self._scalar else self._y.copy()

    def __call__(self, x: ArrayLike) -> float | np.ndarray:
        xa = np.asarray(x, dtype=float)
        if self._y is None or xa.shape != self._y.shape:
            self._scalar = xa.ndim == 0
            self._y = xa.copy()
        else:
            self._y = self._y + self.alpha * (xa - self._y)
        return self.value  # type: ignore[return-value]
