import numpy as np

from substratum.dsp.dynamics import lookahead_limiter, soft_clip

SR = 48000


def test_soft_clip_zero_amount_is_passthrough():
    x = np.linspace(-1, 1, 100)
    assert np.allclose(soft_clip(x, amount=0.0), x)


def test_soft_clip_is_bounded():
    x = np.linspace(-3, 3, 1000)
    assert np.all(np.abs(soft_clip(x, amount=1.0)) <= 1.0)


def test_soft_clip_near_linear_for_small_signals():
    x = np.linspace(-0.05, 0.05, 100)
    y = soft_clip(x, amount=1.0)
    assert np.allclose(y, x, atol=1e-3)


def test_limiter_never_exceeds_threshold():
    x = np.concatenate(
        [
            np.zeros(2000),
            np.full(1000, 2.0),
            np.zeros(2000),
        ]
    )
    out = lookahead_limiter(x, threshold=0.9, sample_rate=SR)
    assert np.max(out) <= 0.9 + 1e-9


def test_limiter_keeps_small_signal_unchanged_in_steady_state():
    x = np.full(5000, 0.2)
    out = lookahead_limiter(x, threshold=0.9, sample_rate=SR)
    tail = out[3000:]
    assert np.allclose(tail, 0.2, atol=1e-3)


def test_limiter_apply_makeup_gain():
    x = np.full(3000, 0.05)
    out = lookahead_limiter(x, threshold=0.9, makeup_db=6.0, sample_rate=SR)
    tail = out[2000:]
    assert np.isclose(np.mean(np.abs(tail)), 0.1, rtol=0.05)


def test_limiter_same_length():
    x = np.random.default_rng(1).normal(size=10000)
    assert len(lookahead_limiter(x, sample_rate=SR)) == len(x)
