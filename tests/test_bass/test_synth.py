import numpy as np

from substratum.bass.synth import SAMPLE_RATE, BassParams, build_voice, decompose_voice, render


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


def test_width_renders_stereo_with_mono_sub():
    audio = render(BassParams(freq=40, width=1.0))
    assert audio.ndim == 2
    assert audio.shape[1] == 2
    left, right = audio[:, 0], audio[:, 1]
    assert not np.allclose(left, right, atol=0.01)
    n = len(left)
    spec_l = np.abs(np.fft.rfft(left * np.hanning(n)))
    spec_r = np.abs(np.fft.rfft(right * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    sub_l = np.max(spec_l[(freqs > 30) & (freqs < 50)])
    sub_r = np.max(spec_r[(freqs > 30) & (freqs < 50)])
    assert np.allclose(sub_l, sub_r, rtol=0.05)


def test_zero_width_stays_mono():
    assert render(BassParams(freq=40)).ndim == 1


def test_tone_hz_overrides_filter():
    low = render(BassParams(freq=40, drive=0.8, tone_hz=100.0))
    high = render(BassParams(freq=40, drive=0.8, tone_hz=600.0))

    def band_energy(audio):
        n = len(audio)
        spec = np.abs(np.fft.rfft(audio * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        mask = (freqs > 150) & (freqs < 550)
        return float(np.sum(spec[mask] ** 2))

    assert band_energy(high) > band_energy(low) * 1.5


def test_sub_level_weakens_fundamental():
    strong = render(BassParams(freq=40, warmth=0.5, sub_level=1.0))
    weak = render(BassParams(freq=40, warmth=0.5, sub_level=0.0))
    n = len(strong)
    spec_s = np.abs(np.fft.rfft(strong * np.hanning(n)))
    spec_w = np.abs(np.fft.rfft(weak * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    fund = np.max(spec_s[(freqs > 30) & (freqs < 50)])
    assert np.max(spec_w[(freqs > 30) & (freqs < 50)]) < fund * 0.2


def test_makeup_changes_output_level():
    plain = render(BassParams(freq=40, drive=0.3))
    hot = render(BassParams(freq=40, drive=0.3, makeup_db=6.0))
    assert not np.allclose(plain, hot, atol=0.01)


def test_curve_blend_is_stable():
    tanh = render(BassParams(freq=40, drive=0.6, curve=0.0))
    atan = render(BassParams(freq=40, drive=0.6, curve=1.0))
    assert not np.allclose(tanh, atan, atol=0.01)


def test_snap_adds_transient():
    quiet = render(BassParams(freq=40, snap=0.0))
    clicked = render(BassParams(freq=40, snap=1.0))
    assert not np.allclose(quiet, clicked, atol=0.01)


def test_distortion_changes_output():
    plain = render(BassParams(freq=40, drive=0.3))
    gritty = render(BassParams(freq=40, drive=0.3, distortion=1.0))
    assert not np.allclose(plain, gritty, atol=0.01)


def test_crush_changes_output():
    plain = render(BassParams(freq=40, drive=0.3))
    crushed = render(BassParams(freq=40, drive=0.3, crush=1.0))
    assert not np.allclose(plain, crushed, atol=0.01)


def test_validation_clamps_new_params():
    p = BassParams(
        decay=99,
        attack=99,
        sustain=9,
        tone_hz=1,
        sub_level=9,
        snap=9,
        width=9,
        curve=9,
        glide=9,
        makeup_db=99,
        distortion=9,
        crush=-9,
    ).validated()
    assert p.decay <= 5.0
    assert p.attack <= 0.5
    assert p.sustain <= 1.0
    assert p.tone_hz == 100.0
    assert p.sub_level <= 1.0
    assert p.width <= 1.0
    assert p.makeup_db <= 12.0
    assert p.curve <= 1.0
    assert p.distortion <= 1.0
    assert p.crush >= 0.0


def test_decompose_voice_layers_sum_to_center():
    p = BassParams(snap=0.5, warmth=0.6)
    parts, end_phase = decompose_voice(p, 40.0, 1.0)
    assert np.allclose(parts.center, parts.sub + parts.snap)
    assert np.allclose(parts.side, parts.warmth)
    assert parts.envelope.shape == parts.sub.shape
    assert np.all(parts.envelope >= 0.0)
    assert isinstance(end_phase, float)


def test_decompose_voice_matches_build_voice():
    p = BassParams(snap=0.3, warmth=0.5, glide=0.4)
    parts, _ = decompose_voice(p, 40.0, 1.0)
    (center, side), _ = build_voice(p, 40.0, 1.0)
    assert np.allclose(parts.center, center)
    assert np.allclose(parts.side, side)


def test_decompose_voice_zero_snap_has_no_transient():
    parts, _ = decompose_voice(BassParams(snap=0.0), 40.0, 0.5)
    assert not np.any(parts.snap)


def test_decompose_voice_zero_duration():
    parts, phase = decompose_voice(BassParams(), 40.0, 0.0)
    assert parts.sub.size == 0
    assert phase == 0.0
