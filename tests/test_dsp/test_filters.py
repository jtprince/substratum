import numpy as np

from substratum.dsp.filters import dc_blocker, one_pole_lowpass

SR = 48000


def test_one_pole_lowpass_attenuates_high_frequency():
    n = SR // 2
    t = np.arange(n) / SR
    high = np.sin(2 * np.pi * 2000 * t)
    out = one_pole_lowpass(high, cutoff_hz=200.0, sample_rate=SR)
    assert np.sqrt(np.mean(out**2)) < np.sqrt(np.mean(high**2)) * 0.3


def test_one_pole_lowpass_passes_low_frequency():
    n = SR // 2
    t = np.arange(n) / SR
    low = np.sin(2 * np.pi * 40 * t)
    out = one_pole_lowpass(low, cutoff_hz=200.0, sample_rate=SR)
    assert np.sqrt(np.mean(out**2)) > np.sqrt(np.mean(low**2)) * 0.8


def test_one_pole_lowpass_same_length():
    x = np.random.default_rng(0).normal(size=1000)
    assert len(one_pole_lowpass(x, 300.0, SR)) == len(x)


def test_dc_blocker_removes_offset():
    x = np.ones(20000) * 0.5
    out = dc_blocker(x)
    assert np.abs(out[-1]) < 1e-3


def test_dc_blocker_preserves_signal_shape_after_transient():
    n = SR
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 40 * t) + 0.1
    out = dc_blocker(x)
    tail = out[int(0.5 * n) :]
    x_tail = x[int(0.5 * n) :]
    assert np.corrcoef(tail, x_tail)[0, 1] > 0.99
