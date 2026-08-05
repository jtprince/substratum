import pytest

from substratum.bass.presets import PRESETS, get_preset, list_presets


def test_list_presets_matches_presets():
    assert set(list_presets()) == set(PRESETS)


def test_get_preset_returns_copy():
    p = get_preset("velvet")
    p.freq = 999.0
    assert get_preset("velvet").freq != 999.0


def test_presets_within_valid_ranges():
    for p in PRESETS.values():
        assert 25.0 <= p.freq <= 70.0
        for knob in (p.punch, p.drive, p.warmth, p.weight):
            assert 0.0 <= knob <= 1.0


def test_earthquake_is_most_extreme():
    eq = get_preset("earthquake")
    assert eq.weight == 1.0
    assert eq.punch == 1.0


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        get_preset("nope")
