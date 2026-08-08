import pytest

from substratum.pattern.notation import (
    NotationError,
    Note,
    parse_note,
    parse_pattern,
    total_beats,
)


def test_default_note_is_sub_bass_c1():
    note = parse_note("C")
    assert note.midi == 24
    assert 32.0 < note.frequency < 33.0


def test_octave_marks_shift():
    assert parse_note("C'").midi == 36
    assert parse_note("C,").midi == 12
    assert parse_note("C''").midi == 48


def test_accidentals():
    assert parse_note("F#").midi == 30
    assert parse_note("Bb").midi == 34


def test_length_in_beats():
    assert parse_note("C2").beats == 2.0
    assert parse_note("E1.5").beats == 1.5
    assert parse_note("G").beats == 1.0


def test_combined_token():
    note = parse_note("F#3")
    assert note.midi == 30
    assert note.beats == 3.0


def test_octave_marker_with_length():
    note = parse_note("F#,3")
    assert note.midi == 18
    assert note.beats == 3.0


def test_pattern_parses_ignoring_bars():
    notes = parse_pattern("C1 E1 G1 | A1")
    assert [n.midi for n in notes] == [24, 28, 31, 33]
    assert total_beats(notes) == 4.0


def test_bad_token_raises():
    with pytest.raises(NotationError):
        parse_pattern("C1 Q9")


def test_zero_length_rejected():
    with pytest.raises(NotationError):
        parse_note("C0")


def test_empty_pattern_is_empty_list():
    assert parse_pattern("") == []


def test_rest_is_none_midi():
    assert parse_note("z1") == Note(midi=None, beats=1.0)
    assert parse_note("r2.5") == Note(midi=None, beats=2.5)
    assert parse_note("Z") == Note(midi=None, beats=1.0)


def test_rest_requires_no_accidental_or_octave():
    with pytest.raises(NotationError):
        parse_note("z#")
    with pytest.raises(NotationError):
        parse_note("r'")


def test_mixed_pattern_with_rests():
    notes = parse_pattern("C2 z1 E1 | G1 r2")
    assert [n.is_rest for n in notes] == [False, True, False, False, True]
    assert total_beats(notes) == 7.0


def test_rest_rejected_zero_length():
    with pytest.raises(NotationError):
        parse_note("z0")
