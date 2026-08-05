from substratum.bass.analyze import analyze, estimate_fundamental, harmonic_series
from substratum.bass.synth import BassParams, render
from substratum.io.audio import read_wav, write_wav


def _render_wav(tmp_path, **kwargs) -> str:
    path = tmp_path / "sound.wav"
    write_wav(path, render(BassParams(**kwargs)), 48000)
    return str(path)


def test_estimate_fundamental_recovers_pitch(tmp_path):
    path = _render_wav(tmp_path, freq=38.0, drive=0.3)
    audio, sr = read_wav(path)
    f0 = estimate_fundamental(audio, sr)
    assert 30 <= f0 <= 46


def test_harmonic_series_length_and_ordering(tmp_path):
    path = _render_wav(tmp_path, freq=40.0, drive=0.3)
    audio, sr = read_wav(path)
    amps = harmonic_series(audio, sr, fundamental_hz=40.0, num_harmonics=8)
    assert len(amps) == 8
    assert all(0.0 <= a <= 1.0 for a in amps)
    assert amps[0] == 1.0


def test_analyze_writes_five_figures(tmp_path):
    wav = _render_wav(tmp_path, freq=38.0, drive=0.3)
    figures = analyze(wav, tmp_path / "figs")
    assert len(figures) == 5
    for fig in figures:
        assert fig.exists()
        assert fig.suffix == ".png"
