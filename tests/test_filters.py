"""Synthetic tests for mcbio.filters."""
import numpy as np
import pytest

from mcbio.filters import EMA, OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    assert f(3.5, 0.0) == pytest.approx(3.5)
    assert isinstance(f(3.5, 0.03), float)


def test_converges_to_constant():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f(0.0, 0.0)
    y = None
    for i in range(1, 300):          # 10 s at 30 Hz of a step to 1.0
        y = f(1.0, i / 30.0)
    assert y == pytest.approx(1.0, abs=1e-3)


def test_reduces_jitter():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = 0.5 + rng.normal(0.0, 0.05, 600)
    out = np.array([f(x, i / 30.0) for i, x in enumerate(raw)])
    assert np.std(out[60:]) < 0.3 * np.std(raw[60:])


def test_beta_reduces_lag_on_fast_motion():
    t = np.arange(0, 3, 1 / 30.0)
    x = 10.0 * t                      # fast ramp
    slow = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    fast = OneEuroFilter(min_cutoff=0.5, beta=1.0)
    lag_slow = x[-1] - [slow(v, ti) for v, ti in zip(x, t)][-1]
    lag_fast = x[-1] - [fast(v, ti) for v, ti in zip(x, t)][-1]
    assert 0 < lag_fast < 0.5 * lag_slow


def test_vector_and_bad_dt():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
    a = f((1.0, 2.0, 3.0), 1.0)
    assert isinstance(a, np.ndarray) and a.shape == (3,)
    b = f((5.0, 5.0, 5.0), 1.0)       # dt == 0 -> previous estimate
    np.testing.assert_allclose(b, a)
    c = f((5.0, 5.0, 5.0), 0.5)       # dt < 0 -> previous estimate
    np.testing.assert_allclose(c, a)
    d = f((5.0, 5.0, 5.0), 1.1)
    assert np.all(d > a) and np.all(d < 5.0)


def test_reset():
    f = OneEuroFilter()
    f(1.0, 0.0)
    f.reset()
    assert f.value is None
    assert f(7.0, 0.1) == pytest.approx(7.0)


def test_ema():
    e = EMA(alpha=0.5)
    assert e(0.0) == 0.0
    assert e(1.0) == pytest.approx(0.5)
    assert e(1.0) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        EMA(alpha=0.0)
