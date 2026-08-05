from substratum.gallery.generate import _cli_command, _resolve_params, generate
from substratum.gallery.sounds import GALLERY_SOUNDS


def test_gallery_sounds_have_required_keys():
    for sound in GALLERY_SOUNDS:
        assert sound["name"]
        assert sound["title"]
        assert sound["description"]
        assert sound.get("preset") or sound.get("params")


def test_cli_command_for_preset_sound():
    sound = next(s for s in GALLERY_SOUNDS if s.get("preset") == "velvet")
    assert _cli_command(sound) == "bass --preset velvet --output warm-velvet.wav"


def test_cli_command_for_param_sound():
    sound = {
        "name": "x",
        "params": {"freq": 45.0, "punch": 0.7, "drive": 0.35, "warmth": 0.2, "weight": 0.4},
    }
    assert "bass --freq 45" in _cli_command(sound)
    assert "--punch 0.7" in _cli_command(sound)


def test_resolve_params_applies_overrides():
    sound = {"name": "x", "params": {"freq": 50.0}}
    p = _resolve_params(sound)
    assert p.freq == 50.0
    assert p.punch == 0.5  # default


def test_generate_builds_gallery(tmp_path):
    index = generate(tmp_path, duration=0.3)
    assert index.exists()
    assert (tmp_path / "sounds").exists()
    mp3s = list((tmp_path / "sounds").glob("*.mp3"))
    assert len(mp3s) == len(GALLERY_SOUNDS)
    html = index.read_text()
    assert "audio" in html
    assert "bass --preset" in html
