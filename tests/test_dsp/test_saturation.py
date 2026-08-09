import numpy as np

from substratum.dsp.saturation import (
    atan_saturate,
    bitcrush,
    downsample,
    foldback_saturate,
    oversample,
    saturate,
    tanh_saturate,
)


def test_oversample_length_and_preserves_shape():
    x = np.linspace(-1, 1, 1000)
    y = oversample(x, 4)
    assert len(y) == 4000
    assert np.isclose(y[0], x[0], atol=1e-2)


def test_downsample_recovers_length():
    x = np.linspace(-1, 1, 4000)
    y = downsample(x, 4)
    assert len(y) == 1000


def test_resample_roundtrip():
    sr = 48000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 40 * t)
    y = downsample(oversample(x, 4), 4)
    assert len(y) == len(x)
    mid = slice(2000, -2000)
    assert np.allclose(y[mid], x[mid], atol=0.02)


def test_tanh_saturate_zero_drive_is_passthrough():
    x = np.linspace(-1, 1, 100)
    assert np.allclose(tanh_saturate(x, 0.0), x)


def test_tanh_saturate_is_bounded():
    x = np.linspace(-5, 5, 1000)
    y = tanh_saturate(x, 1.0)
    assert np.all(np.abs(y) <= 1.0)


def test_tanh_saturate_is_odd_symmetric():
    x = np.linspace(0, 2, 100)
    assert np.allclose(tanh_saturate(x, 0.5), -tanh_saturate(-x, 0.5))


def test_atan_saturate_is_bounded():
    x = np.linspace(-5, 5, 1000)
    y = atan_saturate(x, 1.0)
    assert np.all(np.abs(y) <= 1.0)


def test_tanh_generates_harmonics():
    x = np.sin(2 * np.pi * 40 * np.arange(4800) / 48000)
    y = saturate(x, drive=1.0)
    assert not np.allclose(y, x)


def test_foldback_zero_amount_is_passthrough():
    x = np.linspace(-1, 1, 100)
    assert np.allclose(foldback_saturate(x, 0.0), x)


def test_foldback_is_bounded():
    x = np.linspace(-5, 5, 1000)
    y = foldback_saturate(x, 1.0)
    assert np.all(np.abs(y) <= 1.0)


def test_foldback_generates_harmonics():
    x = np.sin(2 * np.pi * 40 * np.arange(4800) / 48000)
    y = foldback_saturate(x, 1.0)
    assert not np.allclose(y, x)


def test_bitcrush_zero_amount_is_passthrough():
    x = np.linspace(-1, 1, 100)
    assert np.allclose(bitcrush(x, 0.0), x)


def test_bitcrush_is_bounded():
    x = np.linspace(-5, 5, 1000)
    y = bitcrush(x, 1.0)
    assert np.all(np.abs(y) <= 1.0)


def test_bitcrush_quantizes_to_few_levels():
    x = np.linspace(-1, 1, 1000)
    y = bitcrush(x, 1.0)
    assert len(np.unique(np.round(y, 6))) <= 9
