import numpy as np
import pytest

from substratum.bass.synth import BassParams
from substratum.io.samples import (
    SaveError,
    decode_audio,
    delete,
    list_saves,
    load,
    save,
)


def test_save_round_trip_mp3(tmp_path):
    params = BassParams(drive=0.3, warmth=0.5, width=0.6, glide=0.8, snap=0.2, makeup_db=3.0)
    audio_path = save("super_low_punchy", params, "C1 E1 G1", 70, gain=0.9, dir_path=tmp_path)
    assert audio_path.name == "super_low_punchy.mp3"
    assert audio_path.exists()
    assert (tmp_path / "super_low_punchy.json").exists()

    data = load("super_low_punchy", tmp_path)
    assert data["pattern"] == "C1 E1 G1"
    assert data["bpm"] == 70
    assert data["params"].drive == 0.3
    assert data["params"].width == 0.6

    names = [s["name"] for s in list_saves(tmp_path)]
    assert "super_low_punchy" in names

    audio, sr = decode_audio("super_low_punchy", tmp_path)
    assert sr == 48000
    assert audio.ndim == 2
    assert audio.shape[1] == 2
    assert np.max(np.abs(audio)) <= 1.0


def test_save_wav_variant(tmp_path):
    path = save("x", BassParams(), "C1", 70, dir_path=tmp_path, ext="wav")
    assert path.suffix == ".wav"


def test_save_rejects_bad_name(tmp_path):
    with pytest.raises(SaveError):
        save("../evil", BassParams(), "C1", 70, dir_path=tmp_path)


def test_delete_removes_audio_and_json(tmp_path):
    save("delme", BassParams(), "C1", 70, dir_path=tmp_path)
    assert delete("delme", tmp_path)
    assert list_saves(tmp_path) == []


def test_load_missing_raises(tmp_path):
    with pytest.raises(SaveError):
        load("nope", tmp_path)
