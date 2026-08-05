import numpy as np

from substratum.bass.synth import SAMPLE_RATE, BassParams, render


def test_render_default_length():
    audio = render(BassParams(duration=1.0))
    assert len(audio) == SAMPLE_RATE
    assert audio.dtype == np.float64


def test_render_is_normalized_within_headroom():
    audio = render(BassParams())
    assert np.max(np.abs(audio)) <= 1.0
    assert np.max(np.abs(audio)) >= 0.5


def test_render_has_no_dc_offset():
    audio = render(BassParams())
    assert np.abs(np.mean(audio)) < 1e-4


def test_clean_sub_is_nearly_pure_sine():
    p = BassParams(freq=40.0, punch=0.0, drive=0.0, warmth=0.0, weight=0.2)
    audio = render(p)
    n = len(audio)
    spec = np.abs(np.fft.rfft(audio * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    fundamental = np.max(spec[(freqs > 30) & (freqs < 50)])
    second = np.max(spec[(freqs > 70) & (freqs < 90)])
    assert second < fundamental * 0.05


def test_higher_drive_adds_harmonics():
    clean = render(BassParams(freq=40, punch=0, drive=0, warmth=0, weight=0.2))
    driven = render(BassParams(freq=40, punch=0, drive=1.0, warmth=0, weight=0.2))
    assert not np.allclose(clean, driven, atol=0.01)


def test_higher_punch_changes_attack_spectrum():
    no_punch = render(BassParams(freq=40, punch=0.0, drive=0.3, weight=0.2))
    punch = render(BassParams(freq=40, punch=1.0, drive=0.3, weight=0.2))
    assert not np.allclose(no_punch, punch, atol=0.01)


def test_weight_changes_decay_shape():
    lean = render(BassParams(freq=40, punch=0.3, drive=0.3, weight=0.0))
    huge = render(BassParams(freq=40, punch=0.3, drive=0.3, weight=1.0))
    assert not np.allclose(lean, huge, atol=0.01)


def test_validation_clamps_out_of_range():
    p = BassParams(freq=500, punch=5, drive=-3, warmth=9, weight=-1).validated()
    assert p.freq <= 70.0
    assert p.punch <= 1.0
    assert p.drive >= 0.0
    assert p.warmth <= 1.0
    assert p.weight >= 0.0


def test_warmth_above_zero_is_not_buzzy():
    audio = render(BassParams(freq=40, warmth=0.8, drive=0.2))
    n = len(audio)
    spec = np.abs(np.fft.rfft(audio * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    high_band = np.max(spec[freqs > 1000])
    total = np.max(spec)
    assert high_band < total * 0.1
