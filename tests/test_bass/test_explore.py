import pytest

from substratum.bass.explore import PARAM_NAMES, sweep


def test_sweep_writes_expected_number_of_files(tmp_path):
    files = sweep(tmp_path, freqs=(38.0,), step=0.5)
    expected = len(PARAM_NAMES) * 3  # 3 values per param at step 0.5
    assert len(files) == expected
    assert all(f.exists() for f in files)


def test_sweep_naming(tmp_path):
    files = sweep(tmp_path, freqs=(38.0,), step=1.0, params=("drive",))
    assert files[0].name == "freq38_drive00.wav"
    assert files[-1].name == "freq38_drive100.wav"


def test_sweep_files_are_valid_wav(tmp_path):
    import wave

    files = sweep(tmp_path, freqs=(38.0,), step=1.0, params=("punch",))
    with wave.open(str(files[0]), "rb") as w:
        assert w.getframerate() == 48000
        assert w.getsampwidth() == 3  # 24-bit


def test_sweep_rejects_unknown_param(tmp_path):
    with pytest.raises(ValueError):
        sweep(tmp_path, freqs=(38.0,), params=("bogus",))
